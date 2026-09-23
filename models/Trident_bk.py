from unittest import skip
from sympy.logic import false
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import yaml
from copy import deepcopy
from torch.nn.utils import skip_init
import numpy as np

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
        activation_expectation: float = 1.64,  # Expected value of SiLU activation function
        device = "cpu"
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_grids = num_grids
        self.dropout = nn.Dropout(dropout)

        # Calculate the variance of weights
        var_w = 1.0 / (input_dim * activation_expectation)

        # Initialize frequency matrix as learnable parameters using normal distribution
        self.weight = nn.Parameter(torch.randn(input_dim, num_grids, device=device) * math.sqrt(var_w))
        
        # Initialize bias with uniform distribution [0, 2π]
        self.bias = nn.Parameter(torch.empty(num_grids, device=device))
        nn.init.uniform_(self.bias, 0, 2 * 3.14)

        # Map to input_dim
        self.combination = nn.Linear(2 * num_grids, input_dim, device=device)
        
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
        device = "cpu",
        infer = False
    ) -> None:
        super().__init__()
        self.base_activation = base_activation
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.res_net = res_net
        self.use_layernorm = use_layernorm

        self.layernorm = nn.LayerNorm(input_dim, device=device) if use_layernorm and input_dim > 1 else None

        if res_net:
            self.shortcut = nn.Linear(input_dim, output_dim, device=device)

        # 修改特征变换，使其输出维度与 input_dim 匹配
        # print("use rff")
        self.feature_transform = RandomFourierFeatures(
            input_dim=input_dim, 
            num_grids=num_grids, 
            dropout=dropout_rate,
            activation_expectation=activation_expectation,
            device=device
        )

                # Layer 中
        # 初始化可学习的缩放参数
        self.base_scale = nn.Parameter(torch.tensor(1.0, device=device))  # 初始化为1
        self.spline_scale = nn.Parameter(torch.tensor(1e-2, device=device))  # 初始化为小量

        # 不再使用单独的 spline_linear 和 base_linear，而是统一使用一个 final_linear
        self.final_linear = nn.Linear(input_dim, output_dim, device=device)

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
            # del b, s
            ret = self.final_linear(combined)
            # del combined
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
        device = "cpu"
    ):
        super().__init__()
        grid = torch.linspace(grid_min, grid_max, num_grids, device=device)
        self.train_grid = torch.tensor(train_grid, dtype=torch.bool, device=device)
        self.train_inv_denominator = torch.tensor(train_inv_denominator, dtype=torch.bool, device=device) 
        self.grid = torch.nn.Parameter(grid, requires_grad=train_grid)
        #print(f"grid initial shape: {self.grid.shape }")
        self.inv_denominator = torch.nn.Parameter(torch.tensor(inv_denominator, dtype=torch.float32, device=device), requires_grad=train_inv_denominator)  # Cache the inverse of the denominator

    def forward(self, x):
        return RSWAFFunction.apply(x, self.grid, self.inv_denominator, self.train_grid, self.train_inv_denominator)

class ChebyKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree, spline_weight_init_scale, dropout_rate=0.2, res_net=False, device="cpu", infer=False):
        super(ChebyKANLayer, self).__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.degree = degree
        self.res_net = res_net

        if res_net:
            self.shortcut = nn.Linear(input_dim, output_dim, device=device)
            self.layernorm = nn.LayerNorm(input_dim, device=device) if input_dim > 1 else None

        self.dropout = nn.Dropout(dropout_rate)
        self.spline_linear = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1, device=device))
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
        super().__init__(in_features, out_features, bias=False, device=device, **kw)
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
        device = 'cpu',
        infer = False

    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        self.layernorm = None
        self.res_net = res_net

        if self.res_net:
            if infer:
                self.shortcut = skip_init(nn.Linear, input_dim, output_dim, device=device)
            else:
                self.shortcut = nn.Linear(input_dim, output_dim, device=device)
        self.use_layernorm = use_layernorm
        if use_layernorm:
            assert input_dim > 1, "Do not use layernorms on 1D inputs. Set `use_layernorm=False`."
            if infer:
                self.layernorm = skip_init(nn.LayerNorm, input_dim, device=device)
            else:
                self.layernorm = nn.LayerNorm(input_dim, device=device)
        self.rbf = RadialBasisFunction(grid_min, grid_max, num_grids, denominator=denominator, radial_type=radial_type, device=device)
        if radial_type == "Fourier":
            num_grids *= 2
            num_grids += 1
        if infer:
            self.spline_linear = skip_init(SplineLinear, input_dim * num_grids, output_dim, spline_weight_init_scale, device=device)
        else:
            self.spline_linear = SplineLinear(input_dim * num_grids, output_dim, spline_weight_init_scale, device=device)
        self.use_base_update = use_base_update
        if use_base_update:
            self.base_activation = base_activation
            if infer:
                self.base_linear = skip_init(nn.Linear, input_dim, output_dim, device=device)
            else:
                self.base_linear = nn.Linear(input_dim, output_dim, device=device)
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
        # del x, base
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

class CrossLayer(nn.Module):
    def __init__(self, input_dim, infer = False, device = "cpu"):
        
        super().__init__()
        # 参数极少：只有 input_dim 个权重和偏置
        self.w = nn.Parameter(torch.empty(input_dim, 1, device=device))
        self.b = nn.Parameter(torch.empty(input_dim, device=device))
        
        # 初始化
        if not infer:
            nn.init.xavier_uniform_(self.w)
            nn.init.zeros_(self.b)

    def forward(self, x0, xl):
        # x0: 最原始的输入 (batch_size, input_dim)
        # xl: 上一层的输出 (batch_size, input_dim)
        
        # 1. 计算特征的内积标量: (batch_size, input_dim) * (input_dim, 1) -> (batch_size, 1)
        cross_weight = torch.matmul(xl, self.w) 
        
        # 2. 标量广播相乘，加上偏置和残差
        return x0 * cross_weight + self.b + xl

class Stem_Component(nn.Module):
    def __init__(
        self,
        stem_hidden_layer: List[int],
        dropout_rate: float = 0.2,
        use_tanh = False,
        layernorm = False,
        dcn_num = 2,
        device = "cpu",
        infer = False,
     ) -> None:
        super().__init__()
        input_dim = stem_hidden_layer[0]

        self.use_tanh = use_tanh
        self.layernorm = layernorm

        if self.layernorm:
            self.LayerNorm = nn.LayerNorm(input_dim, device=device)

        self.dropout = nn.Dropout(p=dropout_rate)


        self.DCN_weights = nn.ModuleList([CrossLayer(input_dim, infer, device=device) for _ in range(dcn_num)])


        if infer:
            self.MLP_layers = nn.ModuleList([skip_init(nn.Linear, stem_hidden_layer[i], stem_hidden_layer[i+1], device=device) for i in range(len(stem_hidden_layer) - 1)])
            self.Gatings = nn.ModuleList([skip_init(nn.Linear, stem_hidden_layer[i], stem_hidden_layer[i+1], device=device) for i in range(len(stem_hidden_layer) - 1)])
        else:
            self.MLP_layers = nn.ModuleList([nn.Linear(stem_hidden_layer[i], stem_hidden_layer[i+1], device=device) for i in range(len(stem_hidden_layer) - 1)])
            self.Gatings = nn.ModuleList([nn.Linear(stem_hidden_layer[i], stem_hidden_layer[i+1], device=device) for i in range(len(stem_hidden_layer) - 1)])
        
        # 初始化

        self.silu = nn.SiLU()

    def forward(self, input):
        # if self.use_tanh:
        #     input = torch.tanh(input)
        input = self.dropout(input)

        # x = input
        # for i in range(len(self.DCN_weights)):
        #     x = torch.matmul(x, self.DCN_weights[i]) * input + self.DCN_bias[i] + x

        x = self.DCN_weights[0](input, input)
        x = self.DCN_weights[1](input, x)

        x = self.dropout(x)
        for i in range(len(self.MLP_layers)):
            if self.layernorm:
                x = self.LayerNorm(x)
            x = self.MLP_layers[i](x) + self.silu(self.Gatings[i](x))
        return x

class Injector(nn.Module):
    def __init__(
        self,
        input_dim, 
        output_dim,
        grid_min: float = -2.,
        grid_max: float = 2.,
        num_grids = 8,
        use_base_update: bool = True,
        base_activation = F.silu,
        spline_weight_init_scale: float = 0.1,
        denominator = 0.33,
        use_layernorm = True,
        dropout_rate = 0.2,
        radial_type = "Gaussian",
        res_net = False,
        use_tanh = False,
        tanh_norm = 2,
        tanh_denom = 1,
        device = 'cpu',
        infer = False,
    ) -> None:
        super().__init__()
        self.use_tanh = use_tanh
        self.tanh_norm = tanh_norm
        self.tanh_denom = tanh_denom
        self.layers_dim = [input_dim, input_dim *2, input_dim//4, input_dim//2, output_dim*2, output_dim]
        num_grids = [2,4,8,16,16]
        # print(self.layers_dim)
        self.layers = nn.ModuleList([FastKANLayer(
            input_dim = self.layers_dim[i],
            output_dim = self.layers_dim[i+1],
            grid_min = grid_min,
            grid_max = grid_max,
            num_grids = num_grids[i],
            use_base_update=use_base_update,
            base_activation=base_activation,
            spline_weight_init_scale=spline_weight_init_scale,
            denominator=denominator,
            dropout_rate=dropout_rate,
            use_layernorm=use_layernorm,
            radial_type=radial_type,
            res_net=res_net,
            device=device,
            infer=infer,
        ) for i in range(len(self.layers_dim)-1)])


    def forward(self, x):
        for layer in self.layers:
            if self.use_tanh:
                x = self.tanh_norm * torch.tanh(x/self.tanh_denom)
            x = layer(x)
        return x



@MODEL_REGISTRY.register('Trident')
class Trident(nn.Module):
    def __init__(
        self,
        layers_hidden: List[int],
        trident_layers: List[int],
        merge_method = "concat",
        grid_min: float = -2.,
        grid_max: float = 2.,
        num_grids = 8,
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
        tanh_norm = 2,
        tanh_denom = 1,
        device = 'cpu',
        infer = False,
        feature_insert_pos = None,
        feature_insert = None,
        time_choice_pos = None,
        time_choice_feature = None,
        time_choice_method = None,
        time_choice_outdim = 64,
        barra_dim = None,
        trident_heads = None,
    ) -> None:
        super().__init__()
        if use_tanh:
            self.use_layernorm = False
        else:
            self.use_layernorm = use_layernorm
        if feature_insert is None:
            feature_num = 0
        else:
            if len(feature_insert.shape) == 1:
                feature_num = [feature_insert.sum()]
            else:
                feature_num = [feature_insert[i].sum() for i in range(len(feature_insert))]
        if not isinstance(feature_insert_pos, list):
            if feature_insert_pos is None:
                feature_insert_pos = []
                feature_insert = np.array([])
            else:
                feature_insert_pos = [feature_insert_pos]
                feature_insert = np.array([feature_insert])
        
        to_remove = ~time_choice_feature
        for insert_feature in feature_insert:
            to_remove &= ~insert_feature

        self.time_choice_pos = None
        self.time_choice_feature = None
        self.time_choice_method = None

        self.feature_insert = deepcopy(feature_insert)
        if time_choice_pos is not None:
            to_add_feature = np.expand_dims(time_choice_feature, axis=0)
            self.feature_insert = np.concatenate([self.feature_insert, to_add_feature], axis=0)
            self.time_choice_pos = time_choice_pos
            self.time_choice_feature = time_choice_feature
            self.time_choice_method = time_choice_method
            if time_choice_method == "linear_modulation":                
                self.time_choice_block = Injector(
                    time_choice_feature.sum(), 
                    layers_hidden[time_choice_pos] * 2, 
                    grid_min=grid_min,
                    grid_max=grid_max,
                    num_grids=8,
                    use_base_update=use_base_update,
                    base_activation=base_activation,
                    spline_weight_init_scale=spline_weight_init_scale,
                    denominator=0.33,
                    use_layernorm=use_layernorm,
                    dropout_rate=dropout_rate,
                    radial_type=radial_type,
                    res_net=res_net,
                    use_tanh=use_tanh,
                    tanh_norm=tanh_norm,
                    tanh_denom=tanh_denom,
                    device=device,
                    infer=infer,
                    )
                self.lm_dim = layers_hidden[time_choice_pos]
            elif time_choice_method == "barra_estimator":
                self.barra_dim=barra_dim
                self.time_choice_pos = None                
                self.time_choice_block = Injector(
                    time_choice_feature.sum(),
                    self.barra_dim,
                    grid_min=grid_min,
                    grid_max=grid_max,
                    num_grids=8,
                    use_base_update=use_base_update,
                    base_activation=base_activation,
                    spline_weight_init_scale=spline_weight_init_scale,
                    denominator=0.33,
                    use_layernorm=use_layernorm,
                    dropout_rate=dropout_rate,
                    radial_type=radial_type,
                    res_net=res_net,
                    use_tanh=use_tanh,
                    tanh_norm=tanh_norm,
                    tanh_denom=tanh_denom,
                    device=device,
                    infer=infer,
                    )
            else:
                self.time_choice_method=time_choice_method="default"
                # if time_choice_dim not in feature_insert_pos:
                #     feature_insert_pos += [time_choice_pos]
                #     feature_num += [time_choice_outdim]
                # feature_insert = np.concatenate([feature_insert, to_add_feature], axis=0)
                self.time_choice_block = Injector(
                    time_choice_feature.sum(),
                    time_choice_outdim, 
                    grid_min=grid_min,
                    grid_max=grid_max,
                    num_grids=8,
                    use_base_update=use_base_update,
                    base_activation=base_activation,
                    spline_weight_init_scale=spline_weight_init_scale,
                    denominator=0.33,
                    use_layernorm=use_layernorm,
                    dropout_rate=dropout_rate,
                    radial_type=radial_type,
                    res_net=res_net,
                    use_tanh=use_tanh,
                    tanh_norm=tanh_norm,
                    tanh_denom=tanh_denom,
                    device=device,
                    infer=infer,
                    )
        else:
            self.time_choice_method="default"

        # self.feature_insert = feature_insert
        self.feature_insert_pos = feature_insert_pos

        if isinstance(num_grids, int):
            self.num_grids = [num_grids] * (len(layers_hidden) + len(trident_layers) - 1)
        if isinstance(num_grids, list):
            if len(num_grids) != len(layers_hidden) + len(trident_layers) - 1:
                raise ValueError("If grid number is entered in list form, the length must be equal to the number of hidden layers")
            else:
                self.num_grids = num_grids

        if isinstance(denominator, list):
            if len(denominator) != len(layers_hidden) + len(trident_layers) - 1:
                raise ValueError("If denominator is entered in list form, the length must be equal to the number of hidden layers")
            else:
                self.denominator = denominator
        else:
            self.denominator = [denominator] * (len(layers_hidden) + len(trident_layers) - 1)

        self.spear_heads = nn.ModuleList()
        if merge_method == "EW_ensemble":
            hidden_in_dim = trident_layers[-1]
        else:
            hidden_in_dim = 0
        trident_len = len(trident_layers) - 1
        for head in trident_heads:
            head_dim = (head & to_remove).sum()
            if merge_method == "EW_ensemble":
                trident_dim = [math.ceil(head_dim * layer_mult) for layer_mult in trident_layers[:-1]]
                trident_dim += [trident_layers[-1]]
            else:
                trident_dim = [math.ceil(head_dim * layer_mult) for layer_mult in trident_layers]
            self.spear_heads.append(nn.ModuleList([FastKANLayer(
                        trident_dim[j], trident_dim[j+1],
                        grid_min=grid_min,
                        grid_max=grid_max,
                        num_grids=self.num_grids[j],
                        use_base_update=use_base_update,
                        base_activation=base_activation,
                        spline_weight_init_scale=spline_weight_init_scale,
                        denominator=self.denominator[j],
                        dropout_rate=dropout_rate,
                        use_layernorm=self.use_layernorm,
                        radial_type=radial_type,
                        res_net=res_net,
                        device=device,
                        infer=infer,
                    ) for j in range(trident_len)]))
            if merge_method != "EW_ensemble":
                hidden_in_dim += trident_dim[-1]

        self.trident_len = trident_len
        self.trident_heads = [trident_head & to_remove for trident_head in trident_heads]
        self.num_heads = len(self.trident_heads)
        self.merge_method = merge_method
        
        layers_hidden = [hidden_in_dim] + layers_hidden
        if not isinstance(feature_insert_pos, list):
            if mode == "Cheby":
                self.layers = nn.ModuleList([
                    ChebyKANLayer(layers_hidden[i]+feature_num*(i==feature_insert_pos), layers_hidden[i+1], self.num_grids[trident_len+i], spline_weight_init_scale, dropout_rate, res_net, device, infer)
                    for i in range(len(layers_hidden) - 1)
                ])
            elif mode == "KAF":
                self.layers = nn.ModuleList([
                    FastKAFLayer(layers_hidden[i]+feature_num*(i==feature_insert_pos), layers_hidden[i+1], self.num_grids[trident_len+i], self.use_layernorm, dropout_rate, base_activation, spline_weight_init_scale, res_net, device, infer)
                    for i in range(len(layers_hidden) - 1)
                ])
            else:
                self.layers = nn.ModuleList([
                    FastKANLayer(
                        layers_hidden[i]+feature_num*(i==feature_insert_pos), layers_hidden[i+1],
                        grid_min=grid_min,
                        grid_max=grid_max,
                        num_grids=self.num_grids[trident_len+i],
                        use_base_update=use_base_update,
                        base_activation=base_activation,
                        spline_weight_init_scale=spline_weight_init_scale,
                        denominator=self.denominator[trident_len+i],
                        dropout_rate=dropout_rate,
                        use_layernorm=self.use_layernorm,
                        radial_type=radial_type,
                        res_net=res_net,
                        device=device,
                        infer=infer,
                    ) for i in range(len(layers_hidden) - 1)
                ])
        else:
            if mode == "Cheby":
                self.layers = nn.ModuleList([
                    (ChebyKANLayer(layers_hidden[i], layers_hidden[i+1], self.num_grids[trident_len+i], spline_weight_init_scale, dropout_rate, res_net, device, infer) if i not in feature_insert_pos else None)
                    for i in range(len(layers_hidden) - 1)
                ])
                for i in range(len(feature_insert_pos)):
                    self.layers[feature_insert_pos[i]-trident_len] = ChebyKANLayer(layers_hidden[feature_insert_pos[i]-trident_len] + feature_num[i], layers_hidden[feature_insert_pos[i]+1], self.num_grids[feature_insert_pos[i]], spline_weight_init_scale, dropout_rate, res_net, device, infer)
            elif mode == "KAF":
                self.layers = nn.ModuleList([
                    (FastKAFLayer(layers_hidden[i], layers_hidden[i+1], self.num_grids[i], self.use_layernorm, dropout_rate, base_activation, spline_weight_init_scale, res_net, device, infer) if i not in feature_insert_pos else None)
                    for i in range(len(layers_hidden) - 1)
                ])
                for i in range(len(feature_insert_pos)):
                    self.layers[feature_insert_pos[i]-trident_len] = FastKAFLayer(layers_hidden[feature_insert_pos[i]-trident_len] + feature_num[i], layers_hidden[feature_insert_pos[i]-trident_len+1], self.num_grids[feature_insert_pos[i]], self.use_layernorm, dropout_rate, base_activation, spline_weight_init_scale, res_net, device, infer)
            else:
                self.layers = nn.ModuleList([(
                    FastKANLayer(
                        layers_hidden[i], layers_hidden[i+1],
                        grid_min=grid_min,
                        grid_max=grid_max,
                        num_grids=self.num_grids[trident_len+i],
                        use_base_update=use_base_update,
                        base_activation=base_activation,
                        spline_weight_init_scale=spline_weight_init_scale,
                        denominator=self.denominator[trident_len+i],
                        dropout_rate=dropout_rate,
                        use_layernorm=self.use_layernorm,
                        radial_type=radial_type,
                        res_net=res_net,
                        device=device,
                        infer=infer,
                    ) if (i+trident_len not in feature_insert_pos and (i+trident_len != time_choice_pos or time_choice_method!="default")) else None)
                    for i in range(len(layers_hidden) - 1)
                ])
                for i in range(len(feature_insert_pos)):
                    self.layers[feature_insert_pos[i]-trident_len] = FastKANLayer(
                        layers_hidden[feature_insert_pos[i]-trident_len] + feature_num[i] + (feature_insert_pos[i]==time_choice_pos and time_choice_method=="default")*time_choice_outdim, layers_hidden[feature_insert_pos[i]-trident_len+1],
                        grid_min=grid_min,
                        grid_max=grid_max,
                        num_grids=self.num_grids[feature_insert_pos[i]],
                        use_base_update=use_base_update,
                        base_activation=base_activation,
                        spline_weight_init_scale=spline_weight_init_scale,
                        denominator=self.denominator[feature_insert_pos[i]],
                        dropout_rate=dropout_rate,
                        use_layernorm=self.use_layernorm,
                        radial_type=radial_type,
                        res_net=res_net,
                        device=device,
                        infer=infer,
                    )
                if time_choice_pos not in feature_insert_pos and time_choice_method=="default":
                    self.layers[time_choice_pos-trident_len] = FastKANLayer(
                        layers_hidden[time_choice_pos-trident_len] + time_choice_outdim, layers_hidden[time_choice_pos-trident_len+1],
                        grid_min=grid_min,
                        grid_max=grid_max,
                        num_grids=self.num_grids[time_choice_pos],
                        use_base_update=use_base_update,
                        base_activation=base_activation,
                        spline_weight_init_scale=spline_weight_init_scale,
                        denominator=self.denominator[time_choice_pos],
                        dropout_rate=dropout_rate,
                        use_layernorm=self.use_layernorm,
                        radial_type=radial_type,
                        res_net=res_net,
                        device=device,
                        infer=infer,
                    )
            self.feature_insert_pos = np.array(self.feature_insert_pos)
        self.layers_hidden = layers_hidden
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.num_grids = num_grids
        self.use_base_update = use_base_update
        self.base_activation = base_activation
        self.spline_weight_init_scale = spline_weight_init_scale
        self.denominator = denominator
        self.use_tanh = use_tanh
        self.device = device
        self.tanh_norm = tanh_norm
        self.tanh_denom = tanh_denom

    def forward(self, x):
        if self.feature_insert_pos is not None and self.feature_insert is not None:
            if isinstance(self.feature_insert_pos, int):
                feature_insert = x[:, self.feature_insert]
                x = x[:, ~self.feature_insert]
                if self.stem_component:
                    x = self.stem_component.forward(x)
                for i in range(len(self.layers)):
                    if i == self.feature_insert_pos:
                        x = torch.cat([x, feature_insert], dim=1)
                    if self.use_tanh:
                        x = self.tanh_norm * torch.tanh(x/self.tanh_denom)
                    x = self.layers[i](x)
            else:
                if self.time_choice_pos is not None or self.time_choice_method == "barra_estimator":
                    time_choice_feature = x[:, self.time_choice_feature]
                    if self.time_choice_method == "barra_estimator":
                        barra_section = x[..., -self.barra_dim:]
                    time_choice_result = self.time_choice_block(time_choice_feature)
                feature_insert = []
                for features_indices in self.feature_insert:
                    feature_insert.append(x[:, features_indices])
                trident_tips = [x[...,trident_head] for trident_head in self.trident_heads]
                for i in range(self.num_heads):
                    for layer in self.spear_heads[i]:
                        if self.use_tanh:
                            trident_tips[i] = self.tanh_norm * torch.tanh(trident_tips[i]/self.tanh_denom)
                        trident_tips[i] = layer(trident_tips[i])
                if self.merge_method == "EW_ensemble":
                    x = torch.zeros_like(trident_tips[0], dtype=trident_tips[0].dtype, device=trident_tips[0].device)
                    for i in range(self.num_heads):
                        x += trident_tips[i]
                else:
                    x = torch.cat(trident_tips, dim=-1)
                for i in range(len(self.layers)):
                    if self.time_choice_pos is not None and i+self.trident_len == self.time_choice_pos:
                        if self.time_choice_method == "linear_modulation":
                            lm_res = time_choice_result
                            x = x * lm_res[:,:self.lm_dim] + lm_res[:,self.lm_dim:]
                        elif self.time_choice_method == "barra_estimator":
                            x = x
                        else:
                            x = torch.cat([x, time_choice_result], dim=1)
                    if i+self.trident_len in self.feature_insert_pos:
                        idx = np.where(self.feature_insert_pos==i+self.trident_len)[0][0]
                        x = torch.cat([x, feature_insert[idx]], dim=1)
                    if self.use_tanh:
                        x = self.tanh_norm * torch.tanh(x/self.tanh_denom)
                    x = self.layers[i](x)
                    # print(i)
                if self.time_choice_method == "barra_estimator":
                    a = torch.linalg.vecdot(time_choice_result, barra_section, dim=1).unsqueeze(1)
                    # print(self.time_choice_block(time_choice_feature).shape, barra_section.shape, x.shape, a.shape)
                    x += a
            return x

        if self.stem_component:
            x = self.stem_component.forward(x)
        for layer in self.layers:
            if self.use_tanh:
                x = self.tanh_norm * torch.tanh(x/self.tanh_denom)
            x = layer(x)
        return x

'''
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
'''
'''
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
'''
"""
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