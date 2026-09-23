import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import yaml

from typing import *
from tqdm import tqdm
from torch.autograd import Function



from prometheus.utils.utils import WeightedCorrNp
from prometheus.ops import create_op_process
from prometheus.modelpool.basemodel import BaseModel, create_model
from prometheus.utils.registry_factory import TRAINING_REGISTRY, MODEL_REGISTRY

class RandomFourierFeatures(nn.Module):
    def __init__(
        self, 
        input_dim: int,
        num_grids: int, 
        dropout: float = 0.0,  # Dropout probability for Fourier transform
        activation_expectation: float = 1.64  # Expected value of SiLU activation function
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_grids = num_grids
        self.dropout = nn.Dropout(dropout)

        # Calculate the variance of weights
        var_w = 1.0 / (input_dim * activation_expectation)

        # Initialize frequency matrix as learnable parameters using normal distribution
        self.weight = nn.Parameter(torch.randn(input_dim, num_grids) * math.sqrt(var_w))
        
        # Initialize bias with uniform distribution [0, 2π]
        self.bias = nn.Parameter(torch.empty(num_grids))
        nn.init.uniform_(self.bias, 0, 2 * 3.14)

        # Map to input_dim
        self.combination = nn.Linear(2 * num_grids, input_dim)
        
        # Initialize the combination layer weights using Xavier uniform initialization
        # For the bias term, calculate proper bounds based on fan_in and initialize 
        # uniformly within [-1/sqrt(fan_in), 1/sqrt(fan_in)] to maintain variance
        nn.init.xavier_uniform_(self.combination.weight)
        if self.combination.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.combination.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.combination.bias, -bound, bound)

    def forward(self, x):
        projection = torch.matmul(x, self.weight) + self.bias  # (B, num_grids)

        # Fourier transform
        fourier_features = torch.cat(
            [torch.cos(projection), torch.sin(projection)], dim=-1
        )  # (B, 2 * num_grids)
        fourier_features = self.dropout(fourier_features)

        # Map to (B, input_dim)
        output = self.combination(fourier_features)  # (B, input_dim)
        return output

class FastKAFLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_grids: int = 8,
        use_layernorm: bool = True,
        dropout_rate: float = 0,
        base_activation = F.gelu,
        activation_expectation: float = 1.64,
        res_net = False,

    ) -> None:
        super().__init__()
        self.base_activation = base_activation
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.res_net = res_net
        self.use_layernorm = use_layernorm

        self.layernorm = nn.LayerNorm(input_dim) if use_layernorm and input_dim > 1 else None

        if res_net:
            self.shortcut = nn.Linear(input_dim, output_dim)

        # 修改特征变换，使其输出维度与 input_dim 匹配
        # print("use rff")
        self.feature_transform = RandomFourierFeatures(
            input_dim=input_dim, 
            num_grids=num_grids, 
            dropout=dropout_rate,
            activation_expectation=activation_expectation
        )

                # Layer 中
        # 初始化可学习的缩放参数
        self.base_scale = nn.Parameter(torch.tensor(1.0))  # 初始化为1
        self.spline_scale = nn.Parameter(torch.tensor(1e-2))  # 初始化为小量

        # 不再使用单独的 spline_linear 和 base_linear，而是统一使用一个 final_linear
        self.final_linear = nn.Linear(input_dim, output_dim)

        nn.init.xavier_uniform_(self.final_linear.weight)
        if self.final_linear.bias is not None:
            nn.init.zeros_(self.final_linear.bias)

    def forward(self, x):
        # with torch.amp.autocast('cuda'):
            if self.layernorm is not None and self.use_layernorm:
                x_norm = self.layernorm(x)
            else:
                x_norm = x

            # b(x)
            b = self.base_activation(x)

            # spline(x)
            s = self.feature_transform(x_norm)
            # ϕ(x) = W(ab(x) + cs(x))
            combined = self.base_scale * b + self.spline_scale * s
            ret = self.final_linear(combined)
            if self.res_net:
                ret += self.shortcut(x_norm)

            return ret
        
        
class RSWAFFunction(Function):
    @staticmethod
    def forward(ctx, input, grid, inv_denominator, train_grid, train_inv_denominator):
        # Compute the forward pass
        #print('\n')
        #print(f"Forward pass - grid: {(grid[0].item(),grid[-1].item())}, inv_denominator: {inv_denominator.item()}")

        #print(f"grid.shape: {grid.shape }")
        #print(f"grid: {(grid[0],grid[-1]) }")
        #print(f"inv_denominator.shape: {inv_denominator.shape }")
        #print(f"inv_denominator: {inv_denominator }")
        diff = (input[..., None] - grid)
        diff_mul = diff.mul(inv_denominator)
        tanh_diff = torch.tanh(diff)
        tanh_diff_deriviative = -tanh_diff.mul(tanh_diff) + 1  # sech^2(x) = 1 - tanh^2(x)
        
        # Save tensors for backward pass
        ctx.save_for_backward(input, tanh_diff, tanh_diff_deriviative, diff, inv_denominator)
        ctx.train_grid = train_grid
        ctx.train_inv_denominator = train_inv_denominator
        
        return tanh_diff_deriviative

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved tensors
        input, tanh_diff, tanh_diff_deriviative, diff, inv_denominator = ctx.saved_tensors
        grad_grid = None
        grad_inv_denominator = None
        
        #print(f"tanh_diff_deriviative shape: {tanh_diff_deriviative.shape }")
        #print(f"tanh_diff shape: {tanh_diff.shape }")
        #print(f"grad_output shape: {grad_output.shape }")
        
        # Compute the backward pass for the input
        grad_input = -2 * tanh_diff * tanh_diff_deriviative * grad_output
        #print(f"Backward pass 1 - grad_input: {(grad_input.min().item(), grad_input.max().item())}")
        #print(f"grad_input shape: {grad_input.shape }")
        #print(f"grad_input.sum(dim=-1): {grad_input.sum(dim=-1).shape}")
        grad_input = grad_input.sum(dim=-1).mul(inv_denominator)
        #print(f"Backward pass 2 - grad_input: {(grad_input.min().item(), grad_input.max().item())}")
        #print(f"grad_input: {grad_input}")
        #print(f"grad_input shape: {grad_input.shape }")
        
        # Compute the backward pass for grid
        if ctx.train_grid:
            #print('\n')
            #print(f"grad_grid shape: {grad_grid.shape }")
            grad_grid = -inv_denominator * grad_output.sum(dim=0).sum(dim=0)#-(inv_denominator * grad_output * tanh_diff_deriviative).sum(dim=0) #-inv_denominator * grad_output.sum(dim=0).sum(dim=0)
            #print(f"Backward pass - grad_grid: {(grad_grid[0].item(),grad_grid[-1].item())}")
            #print(f"grad_grid.shape: {grad_grid.shape }")
            #print(f"grad_grid: {(grad_grid[0],grad_grid[-1]) }")
            #print(f"inv_denominator shape: {inv_denominator.shape }")
            #print(f"grad_grid shape: {grad_grid.shape }")

        # Compute the backward pass for inv_denominator        
        if ctx.train_inv_denominator:
            grad_inv_denominator = (grad_output* diff).sum() #(grad_output * diff * tanh_diff_deriviative).sum() #(grad_output* diff).sum() 
            #print(f"Backward pass - grad_inv_denominator: {grad_inv_denominator.item()}")
            #print(f"diff shape: {diff.shape }")

            #print(f"grad_inv_denominator shape: {grad_inv_denominator.shape }")
            #print(f"grad_inv_denominator : {grad_inv_denominator }")

        return grad_input, grad_grid, grad_inv_denominator, None, None # same number as tensors or parameters



class ReflectionalSwitchFunction(nn.Module):
    def __init__(
        self,
        grid_min: float = -1.2,
        grid_max: float = 0.2,
        num_grids: int = 8,
        exponent: int = 2,
        inv_denominator: float = 0.5,
        train_grid: bool = False,        
        train_inv_denominator: bool = False,
    ):
        super().__init__()
        grid = torch.linspace(grid_min, grid_max, num_grids)
        self.train_grid = torch.tensor(train_grid, dtype=torch.bool)
        self.train_inv_denominator = torch.tensor(train_inv_denominator, dtype=torch.bool) 
        self.grid = torch.nn.Parameter(grid, requires_grad=train_grid)
        #print(f"grid initial shape: {self.grid.shape }")
        self.inv_denominator = torch.nn.Parameter(torch.tensor(inv_denominator, dtype=torch.float32), requires_grad=train_inv_denominator)  # Cache the inverse of the denominator

    def forward(self, x):
        return RSWAFFunction.apply(x, self.grid, self.inv_denominator, self.train_grid, self.train_inv_denominator)

class ChebyKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree, spline_weight_init_scale, dropout_rate=0.2, res_net=False):
        super(ChebyKANLayer, self).__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.degree = degree
        self.res_net = res_net

        if res_net:
            self.shortcut = nn.Linear(input_dim, output_dim)
            self.layernorm = nn.LayerNorm(input_dim) if input_dim > 1 else None

        self.dropout = nn.Dropout(dropout_rate)
        self.spline_linear = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.normal_(self.spline_linear, mean=0.0, std=1 / (input_dim * (degree + 1)))
        self.register_buffer("arange", torch.arange(0, degree + 1, 1))

    def forward(self, x):
        # Since Chebyshev polynomial is defined in [-1, 1]
        # We need to normalize x to [-1, 1] using tanh
        x = self.dropout(x)
        if self.res_net and self.layernorm is not None:
            shortcut = self.shortcut(self.layernorm(x))

        x = torch.tanh(x)
        # View and repeat input degree + 1 times
        x = x.view((-1, self.inputdim, 1)).expand(
            -1, -1, self.degree + 1
        )  # shape = (batch_size, inputdim, self.degree + 1)
        # Apply acos
        x = x.acos()
        # Multiply by arange [0 .. degree]
        x *= self.arange
        # Apply cos
        x = x.cos()
        # Compute the Chebyshev interpolation
        y = torch.einsum(
            "bid,iod->bo", x, self.spline_linear
        )  # shape = (batch_size, outdim)
        y = y.view(-1, self.outdim)
        if self.res_net:
            y += shortcut
        return y

class SplineLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, init_scale: float = 0.1, device = 'cpu', **kw) -> None:
        self.init_scale = init_scale
        super().__init__(in_features, out_features, bias=False, **kw)
        self.device = device

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.weight, mean=0, std=self.init_scale, a=-1, b=1)

class RadialBasisFunction(nn.Module):
    def __init__(
        self,
        grid_min: float = -2.,
        grid_max: float = 2.,
        num_grids: int = 8,
        denominator: float = None,  # larger denominators lead to smoother basis
        radial_type: str = "Gaussian",
        device = 'cpu'
    ):
        super().__init__()
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.num_grids = num_grids
        self.radial_type = radial_type
        if radial_type == "Fourier" or radial_type == "Cthulu":
            grid = torch.arange(1, num_grids+1, device=device)
        else:
            grid = torch.linspace(grid_min, grid_max, num_grids, device=device)
        self.grid = torch.nn.Parameter(grid, requires_grad=False)
        self.denominator = denominator or (grid_max - grid_min) / (num_grids - 1)

    def forward(self, x):
        if self.radial_type == "Cthulu":
            return torch.cat([torch.sin(x*self.grid), torch.cos(x*self.grid)], dim=-1)
        elif self.radial_type == "Fourier":
            return torch.cat([x[..., None], torch.sin(x[..., None]*self.grid), torch.cos(x[..., None]*self.grid)], dim=-1)
        elif self.radial_type == "Gaussian":
            return torch.exp(-((x[..., None] - self.grid) / self.denominator).pow(2))
        elif self.radial_type == "RSWAF":
            return RSWAFFunction.apply(x, self.grid, self.denominator, False, False)
        elif self.radial_type == "Inverse_Quadratic":
            return 1 / (1 + ((x[..., None] - self.grid) / self.denominator).pow(2))

class FastKANLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        grid_min: float = -2.,
        grid_max: float = 2.,
        num_grids: int = 8,
        use_base_update: bool = True,
        use_layernorm: bool = True,
        base_activation = F.silu,
        spline_weight_init_scale: float = 0.1,
        denominator = 0.33,
        dropout_rate = 0.2,
        radial_type = "Gaussian",
        res_net = False,
        device = 'cpu'

    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        self.layernorm = None
        self.res_net = res_net

        if self.res_net:
            self.shortcut = nn.Linear(input_dim, output_dim)
        self.use_layernorm = use_layernorm
        if use_layernorm:
            assert input_dim > 1, "Do not use layernorms on 1D inputs. Set `use_layernorm=False`."
            self.layernorm = nn.LayerNorm(input_dim)
        self.rbf = RadialBasisFunction(grid_min, grid_max, num_grids, denominator=denominator, radial_type=radial_type)
        if radial_type == "Fourier":
            num_grids *= 2
            num_grids += 1
        self.spline_linear = SplineLinear(input_dim * num_grids, output_dim, spline_weight_init_scale)
        self.use_base_update = use_base_update
        if use_base_update:
            self.base_activation = base_activation
            self.base_linear = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(p=dropout_rate)


    def forward(self, x):
        x = self.dropout(x)
        if self.layernorm is not None and self.use_layernorm:
            x = self.layernorm(x)
        spline_basis = self.rbf(x)
        ret = self.spline_linear(spline_basis.view(*spline_basis.shape[:-2], -1))
        if self.use_base_update:
            base = self.base_linear(self.base_activation(x))
            ret = ret + base
        if self.res_net:
            ret += self.shortcut(x)
        return ret
"""
    def plot_curve(
        self,
        input_index: int,
        output_index: int,
        num_pts: int = 1000,
        num_extrapolate_bins: int = 2
    ):
        '''this function returns the learned curves in a FastKANLayer.
        input_index: the selected index of the input, in [0, input_dim) .
        output_index: the selected index of the output, in [0, output_dim) .
        num_pts: num of points sampled for the curve.
        num_extrapolate_bins (N_e): num of bins extrapolating from the given grids. The curve 
            will be calculate in the range of [grid_min - h * N_e, grid_max + h * N_e].
        '''
        ng = self.rbf.num_grids
        h = self.rbf.denominator
        assert input_index < self.input_dim
        assert output_index < self.output_dim
        w = self.spline_linear.weight[
            output_index, input_index * ng : (input_index + 1) * ng
        ]   # num_grids,
        x = torch.linspace(
            self.rbf.grid_min - num_extrapolate_bins * h,
            self.rbf.grid_max + num_extrapolate_bins * h,
            num_pts
        )   # num_pts, num_grids
        with torch.no_grad():
            y = (w * self.rbf(x.to(w.dtype))).sum(-1)
        return x, y"""


@MODEL_REGISTRY.register('KANUNet')
class FastKAN(nn.Module):
    def __init__(
        self,
        layers_hidden: List[int],
        Unet_pos=5,
        grid_min: float = -2.,
        grid_max: float = 2.,
        num_grids: int = 8,
        use_base_update: bool = True,
        base_activation = F.silu,
        spline_weight_init_scale: float = 0.1,
        denominator = 0.33,
        use_layernorm = True,
        dropout_rate = 0.2,
        radial_type = "Gaussian",
        use_tanh = False,
        mode = "default",
        res_net = False,
        device = 'cpu'
    ) -> None:
        super().__init__()
        if use_tanh:
            use_layernorm = False
        if mode == "Cheby":
            self.layers = nn.ModuleList([
                ChebyKANLayer(layers_hidden[i]+(layers_hidden[0]*(i==Unet_pos)), layers_hidden[i+1], num_grids, spline_weight_init_scale, dropout_rate, res_net)
                for i in range(len(layers_hidden)-1)
            ])
        elif mode == "KAF":
            self.layers = nn.ModuleList([
                FastKAFLayer(layers_hidden[i]+(layers_hidden[0]*(i==Unet_pos)), layers_hidden[i+1], num_grids, use_layernorm, dropout_rate, base_activation, spline_weight_init_scale, res_net)
                for i in range(len(layers_hidden)-1)
            ])
        else:
            self.layers = nn.ModuleList([
                FastKANLayer(
                    layers_hidden[i]+(layers_hidden[0]*(i==Unet_pos)), layers_hidden[i+1],
                    grid_min=grid_min,
                    grid_max=grid_max,
                    num_grids=num_grids,
                    use_base_update=use_base_update,
                    base_activation=base_activation,
                    spline_weight_init_scale=spline_weight_init_scale,
                    denominator=denominator,
                    dropout_rate=dropout_rate,
                    use_layernorm=use_layernorm,
                    radial_type=radial_type,
                    res_net=res_net,
                    device=device
                )
                for i in range(len(layers_hidden)-1)
            ])
        self.layers_hidden = layers_hidden
        self.Unet_pos = Unet_pos
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.num_grids = num_grids
        self.use_base_update = use_base_update
        self.base_activation = base_activation
        self.spline_weight_init_scale = spline_weight_init_scale
        self.denominator = denominator
        self.use_tanh = use_tanh
        self.device = device

    def forward(self, x):
        original_input = x.clone()
        for i, layer in enumerate(self.layers):
            if i == self.Unet_pos:
                x = torch.cat([x, original_input], dim=1)
            if self.use_tanh:
                x = 2 * torch.tanh(x)
            x = layer(x)
        return x

    def fit(self, train_data, train_target, opt="Adam", Epoch=100, log=1, lamb=0, lambl1=1., lamb_entropy=2., lamb_coef=0., lamb_coefdiff=0, loss_fn=None, lr=1, scheduler_factor=0.5, scheduler_patience=5, ckpt_path="/dfs/data/models", batch=500):

        train_data = F.normalize(torch.from_numpy(train_data), p=2.0, dim=0)
        train_target = torch.from_numpy(train_target)

        num_sample = train_data.shape[0]
        steps = num_sample // batch + 1

        pbar = tqdm(range(Epoch), desc="Training", ncols=100)

        if loss_fn == None:
            loss_fn = loss_fn_eval = torch.nn.MSELoss()
        else:
            loss_fn_eval = loss_fn

        if opt == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        elif opt == "AdamW":
            optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        elif opt == "RMSprop":
            optimizer = torch.optim.RMSprop(self.parameters(), lr=lr)
        elif opt == "LBFGS":
            optimizer = torch.optim.LBFGS(self.parameters(), lr=lr, history_size=10, tolerance_grad=1e-32, tolerance_change=1e-32)
        else:
            raise ValueError("opt must be Adam, LBFGS, AdamW or RMSprop")
        
        result = {"train_loss": [], "test_loss": [], "reg": []}
        

        def closure():
            optimizer.zero_grad()
            pred = self.forward(cur_train_data.to(self.device))
            loss = loss_fn_eval(pred, cur_train_target.to(self.device))
            loss.backward()
            return loss

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=scheduler_factor, patience=scheduler_patience)

        for epoch in pbar:
            indices = torch.randperm(num_sample)

            shuffled_data = train_data[indices].to(torch.float32)
            shuffled_target = train_target[indices].to(torch.float32)

            data_minibatches = torch.tensor_split(shuffled_data, steps)
            target_minibatches = torch.tensor_split(shuffled_target, steps)

            eval_data = data_minibatches[-1]
            eval_target = target_minibatches[-1]

            for step in range(steps-1):
                cur_train_data = data_minibatches[step]
                cur_train_target = target_minibatches[step]

                if opt == "LBFGS":
                    train_loss = optimizer.step(closure)
                else:
                    optimizer.zero_grad()                        
                    pred = self.forward(cur_train_data.to(self.device))
                    train_loss = loss_fn_eval(pred, cur_train_target.to(self.device))
                    train_loss.backward()
                    optimizer.step()
            
                result["train_loss"].append(train_loss.item())

            test_loss = loss_fn_eval(self.forward(eval_data.to(self.device)), eval_target.to(self.device))

            pbar.set_description("Epoch: %d | Train Loss: %.2e | Test Loss: %.2e" % (epoch, train_loss.item(), test_loss.item()))

            result["test_loss"].append(test_loss.item())
            scheduler.step(test_loss)

    def save_checkpoint(self, path):
        model = self

        dic = dict(
            layers_hidden=model.layers_hidden,
            grid_min=model.grid_min,
            grid_max=model.grid_max,
            num_grids=model.num_grids,
            use_base_update=model.use_base_update,
            spline_weight_init_scale=model.spline_weight_init_scale,
            denominator=model.denominator,
            device=str(model.device),
        )

        # Note: base_activation is a function, cannot be serialized directly
        # We'll store its name instead
        if hasattr(model.base_activation, '__name__'):
            dic['base_activation_name'] = model.base_activation.__name__
        else:
            dic['base_activation_name'] = 'silu'

        with open(f"{path}_config.yml", "w") as output_file:
            yaml.dump(dic, output_file, default_flow_style=False)

        torch.save(model.state_dict(), f'{path}_state_dict.pth')
        torch.save(model, f'{path}_model.pth')

    def load_checkpoint(self, path):
        with open(f"{path}_config.yml", "r") as input_file:
            dic = yaml.safe_load(input_file)
        
        # Handle base_activation
        if 'base_activation_name' in dic:
            activation_name = dic.pop('base_activation_name')
            if activation_name == 'silu':
                dic['base_activation'] = F.silu
            elif activation_name == 'gelu':
                dic['base_activation'] = F.gelu  # default fallback
        else:
            dic['base_activation'] = F.silu
            
        model = FastKAN(**dic)
        model.load_state_dict(torch.load(f'{path}_state_dict.pth', map_location='cpu'))
        return model






"""
class AttentionWithFastKANTransform(nn.Module):
    
    def __init__(
        self,
        q_dim: int,
        k_dim: int,
        v_dim: int,
        head_dim: int,
        num_heads: int,
        gating: bool = True,
    ):
        super(AttentionWithFastKANTransform, self).__init__()

        self.num_heads = num_heads
        total_dim = head_dim * self.num_heads
        self.gating = gating
        self.linear_q = FastKANLayer(q_dim, total_dim)
        self.linear_k = FastKANLayer(k_dim, total_dim)
        self.linear_v = FastKANLayer(v_dim, total_dim)
        self.linear_o = FastKANLayer(total_dim, q_dim)
        self.linear_g = None
        if self.gating:
            self.linear_g = FastKANLayer(q_dim, total_dim)
        # precompute the 1/sqrt(head_dim)
        self.norm = head_dim**-0.5

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        bias: torch.Tensor = None,      # additive attention bias
    ) -> torch.Tensor:

        wq = self.linear_q(q).view(*q.shape[:-1], 1, self.num_heads, -1) * self.norm     # *q1hc
        wk = self.linear_k(k).view(*k.shape[:-2], 1, k.shape[-2], self.num_heads, -1)    # *1khc
        att = (wq * wk).sum(-1).softmax(-2)     # *qkh
        del wq, wk
        if bias is not None:
            att = att + bias[..., None]

        wv = self.linear_v(v).view(*v.shape[:-2],1, v.shape[-2], self.num_heads, -1)     # *1khc
        o = (att[..., None] * wv).sum(-3)        # *qhc
        del att, wv

        o = o.view(*o.shape[:-2], -1)           # *q(hc)

        if self.linear_g is not None:
            # gating, use raw query input
            g = self.linear_g(q)
            o = torch.sigmoid(g) * o

        # merge heads
        o = self.linear_o(o)
        return o
"""