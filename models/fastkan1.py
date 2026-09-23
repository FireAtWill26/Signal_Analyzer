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
from utils.util_funcs import RMSNorm

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
        num_grids = [4,4,8,8,32]
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


class MarketRouter(nn.Module):
    """Route market features to a fixed number of experts."""

    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        hidden_dim: int = 64,
        num_hidden_layers: int = 2,
        dropout_rate: float = 0.0,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("Router input_dim must be positive")
        if num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if num_hidden_layers < 0:
            raise ValueError("router_num_layers must be non-negative")
        if temperature <= 0:
            raise ValueError("router_temperature must be positive")
        if top_k is not None and not 1 <= top_k <= num_experts:
            raise ValueError("router_top_k must be in [1, num_experts]")

        self.num_experts = num_experts
        self.temperature = temperature
        self.top_k = top_k

        modules = [nn.LayerNorm(input_dim, device=device)]
        in_dim = input_dim
        for _ in range(num_hidden_layers):
            modules.extend([
                nn.Linear(in_dim, hidden_dim, device=device),
                nn.SiLU(),
            ])
            if dropout_rate > 0:
                modules.append(nn.Dropout(dropout_rate))
            in_dim = hidden_dim
        modules.append(nn.Linear(in_dim, num_experts, device=device))
        self.network = nn.Sequential(*modules)

    def forward(self, x):
        logits = self.network(x) / self.temperature
        if self.top_k is not None and self.top_k < self.num_experts:
            top_values, top_indices = torch.topk(logits, self.top_k, dim=-1)
            sparse_logits = torch.full_like(logits, float("-inf"))
            logits = sparse_logits.scatter(-1, top_indices, top_values)
        return torch.softmax(logits, dim=-1)


class ParallelExpertFastKANLayer(nn.Module):
    """Vectorized independent FastKAN layers.

    Inputs and outputs retain an explicit expert axis (..., E, D), so no
    activation or parameter is shared between expert branches accidentally.
    The RBF grid and LayerNorm affine parameters are intentionally shared.
    """

    def __init__(
        self,
        num_experts: int,
        input_dim: int,
        output_dim: int,
        grid_min: float = -2.,
        grid_max: float = 2.,
        num_grids: int = 8,
        use_base_update: bool = True,
        base_activation=F.silu,
        spline_weight_init_scale: float = 0.1,
        denominator=0.33,
        dropout_rate: float = 0.2,
        use_layernorm: bool = True,
        radial_type: str = "Gaussian",
        res_net: bool = False,
        device: str = "cpu",
        infer: bool = False,
    ) -> None:
        super().__init__()
        if num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if use_layernorm and input_dim <= 1:
            raise ValueError("Do not use layernorms on 1D expert inputs")

        self.num_experts = num_experts
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_base_update = use_base_update
        self.base_activation = base_activation
        self.res_net = res_net
        self.dropout = nn.Dropout(dropout_rate)
        self.layernorm = (
            nn.LayerNorm(input_dim, device=device) if use_layernorm else None
        )
        self.rbf = RadialBasisFunction(
            grid_min,
            grid_max,
            num_grids,
            denominator=denominator,
            radial_type=radial_type,
            device=device,
        )

        basis_dim = 2 * num_grids + 1 if radial_type == "Fourier" else num_grids
        self.spline_weight = nn.Parameter(torch.empty(
            num_experts,
            output_dim,
            input_dim * basis_dim,
            device=device,
        ))

        if use_base_update:
            self.base_weight = nn.Parameter(torch.empty(
                num_experts, output_dim, input_dim, device=device
            ))
            self.base_bias = nn.Parameter(torch.empty(
                num_experts, output_dim, device=device
            ))

        if res_net:
            self.shortcut_weight = nn.Parameter(torch.empty(
                num_experts, output_dim, input_dim, device=device
            ))
            self.shortcut_bias = nn.Parameter(torch.empty(
                num_experts, output_dim, device=device
            ))

        if not infer:
            self.reset_parameters(spline_weight_init_scale)

    @staticmethod
    def _reset_linear_parameters(weight, bias):
        for expert_weight in weight:
            nn.init.kaiming_uniform_(expert_weight, a=math.sqrt(5))
        if bias is not None:
            bound = 1 / math.sqrt(weight.shape[-1])
            nn.init.uniform_(bias, -bound, bound)

    def reset_parameters(self, spline_weight_init_scale):
        nn.init.trunc_normal_(
            self.spline_weight,
            mean=0,
            std=spline_weight_init_scale,
            a=-1,
            b=1,
        )
        if self.use_base_update:
            self._reset_linear_parameters(self.base_weight, self.base_bias)
        if self.res_net:
            self._reset_linear_parameters(
                self.shortcut_weight, self.shortcut_bias
            )

    def forward(self, x):
        # x: (..., num_experts, input_dim)
        x = self.dropout(x)
        if self.layernorm is not None:
            x = self.layernorm(x)

        spline_basis = self.rbf(x).flatten(start_dim=-2)
        ret = torch.einsum(
            "...ei,eoi->...eo", spline_basis, self.spline_weight
        )

        if self.use_base_update:
            base = torch.einsum(
                "...ei,eoi->...eo",
                self.base_activation(x),
                self.base_weight,
            )
            ret = ret + base + self.base_bias

        if self.res_net:
            shortcut = torch.einsum(
                "...ei,eoi->...eo", x, self.shortcut_weight
            )
            ret = ret + shortcut + self.shortcut_bias
        return ret



@MODEL_REGISTRY.register('FastKAN1')
class FastKAN(nn.Module):
    def __init__(
        self,
        layers_hidden: List[int],
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
        stem_component = False,
        stem_layer = 0,
        dcn_num = 2,
        expert_pos = None,
        num_experts: int = 4,
        router_hidden_dim: int = 64,
        router_num_layers: int = 2,
        router_dropout_rate: float = 0.0,
        router_temperature: float = 1.0,
        router_top_k: Optional[int] = None,
        moe_residual_scale: float = 0.1,
        attn_res_pos = None,
    ) -> None:
        super().__init__()
        if use_tanh:
            self.use_layernorm = False
        else:
            self.use_layernorm = use_layernorm
        self.stem_component = stem_component
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
        
        self.res_net = None
        if not isinstance(res_net, bool):
            self.res_net = res_net
            res_net = False
        
        self.time_choice_pos = None
        self.time_choice_feature = None
        self.time_choice_method = None
        self.moe_enabled = time_choice_method in ("moe", "moe_output")
        self.expert_pos = expert_pos
        self.num_experts = num_experts
        self.moe_residual_scale = moe_residual_scale

        self.feature_insert = deepcopy(feature_insert)
        if time_choice_pos is not None or self.moe_enabled:
            if time_choice_feature is None:
                raise ValueError(
                    "time_choice_feature is required when time choice is enabled"
                )
            to_add_feature = np.expand_dims(time_choice_feature, axis=0)
            if self.feature_insert.size == 0:
                self.feature_insert = to_add_feature
            else:
                self.feature_insert = np.concatenate(
                    [self.feature_insert, to_add_feature], axis=0
                )
            self.time_choice_pos = time_choice_pos
            self.time_choice_feature = time_choice_feature
            self.time_choice_method = time_choice_method
            if self.moe_enabled:
                if expert_pos is None:
                    raise ValueError(
                        "expert_pos is required when time_choice_method is 'moe'"
                    )
                if mode != "default":
                    raise ValueError(
                        "MoE currently supports mode='default' FastKANLayer only"
                    )
                self.time_choice_pos = None
                self.time_choice_block = MarketRouter(
                    input_dim=int(time_choice_feature.sum()),
                    num_experts=num_experts,
                    hidden_dim=router_hidden_dim,
                    num_hidden_layers=router_num_layers,
                    dropout_rate=router_dropout_rate,
                    temperature=router_temperature,
                    top_k=router_top_k,
                    device=device,
                )
            elif time_choice_method == "linear_modulation":
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


        if stem_component and stem_layer:
            stem_hidden_layer = layers_hidden[:stem_layer]
            layers_hidden = layers_hidden[stem_layer-1:]
            self.stem_component = Stem_Component(stem_hidden_layer, dropout_rate, use_tanh, use_layernorm, dcn_num, device, infer)

        if isinstance(num_grids, int):
            self.num_grids = [num_grids] * (len(layers_hidden) - 1)
        if isinstance(num_grids, list):
            if len(num_grids) != len(layers_hidden) - 1:
                raise ValueError("If grid number is entered in list form, the length must be equal to the number of hidden layers")
            else:
                self.num_grids = num_grids

        if isinstance(denominator, list):
            if len(denominator) != len(layers_hidden) - 1:
                raise ValueError("If denominator is entered in list form, the length must be equal to the number of hidden layers")
            else:
                self.denominator = denominator
        else:
            self.denominator = [denominator] * (len(layers_hidden) - 1)

        if not self.moe_enabled:
            if not isinstance(feature_insert_pos, list):
                if mode == "Cheby":
                    self.layers = nn.ModuleList([
                        ChebyKANLayer(layers_hidden[i]+feature_num*(i==feature_insert_pos), layers_hidden[i+1], self.num_grids[i], spline_weight_init_scale, dropout_rate, res_net, device, infer)
                        for i in range(len(layers_hidden) - 1)
                    ])
                elif mode == "KAF":
                    self.layers = nn.ModuleList([
                        FastKAFLayer(layers_hidden[i]+feature_num*(i==feature_insert_pos), layers_hidden[i+1], self.num_grids[i], self.use_layernorm, dropout_rate, base_activation, spline_weight_init_scale, res_net, device, infer)
                        for i in range(len(layers_hidden) - 1)
                    ])
                else:
                    self.layers = nn.ModuleList([
                        FastKANLayer(
                            layers_hidden[i]+feature_num*(i==feature_insert_pos), layers_hidden[i+1],
                            grid_min=grid_min,
                            grid_max=grid_max,
                            num_grids=self.num_grids[i],
                            use_base_update=use_base_update,
                            base_activation=base_activation,
                            spline_weight_init_scale=spline_weight_init_scale,
                            denominator=self.denominator[i],
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
                        (ChebyKANLayer(layers_hidden[i], layers_hidden[i+1], self.num_grids[i], spline_weight_init_scale, dropout_rate, res_net, device, infer) if i not in feature_insert_pos else None)
                        for i in range(len(layers_hidden) - 1)
                    ])
                    for i in range(len(feature_insert_pos)):
                        self.layers[feature_insert_pos[i]] = ChebyKANLayer(layers_hidden[feature_insert_pos[i]] + feature_num[i], layers_hidden[feature_insert_pos[i]+1], self.num_grids[feature_insert_pos[i]], spline_weight_init_scale, dropout_rate, res_net, device, infer)
                elif mode == "KAF":
                    self.layers = nn.ModuleList([
                        (FastKAFLayer(layers_hidden[i], layers_hidden[i+1], self.num_grids[i], self.use_layernorm, dropout_rate, base_activation, spline_weight_init_scale, res_net, device, infer) if i not in feature_insert_pos else None)
                        for i in range(len(layers_hidden) - 1)
                    ])
                    for i in range(len(feature_insert_pos)):
                        self.layers[feature_insert_pos[i]] = FastKAFLayer(layers_hidden[feature_insert_pos[i]] + feature_num[i], layers_hidden[feature_insert_pos[i]+1], self.num_grids[feature_insert_pos[i]], self.use_layernorm, dropout_rate, base_activation, spline_weight_init_scale, res_net, device, infer)
                else:
                    self.layers = nn.ModuleList([(
                        FastKANLayer(
                            layers_hidden[i], layers_hidden[i+1],
                            grid_min=grid_min,
                            grid_max=grid_max,
                            num_grids=self.num_grids[i],
                            use_base_update=use_base_update,
                            base_activation=base_activation,
                            spline_weight_init_scale=spline_weight_init_scale,
                            denominator=self.denominator[i],
                            dropout_rate=dropout_rate,
                            use_layernorm=self.use_layernorm,
                            radial_type=radial_type,
                            res_net=res_net,
                            device=device,
                            infer=infer,
                        ) if (i not in feature_insert_pos and (i != time_choice_pos or time_choice_method!="default")) else None)
                        for i in range(len(layers_hidden) - 1)
                    ])
                    for i in range(len(feature_insert_pos)):
                        self.layers[feature_insert_pos[i]] = FastKANLayer(
                            layers_hidden[feature_insert_pos[i]] + feature_num[i] + (feature_insert_pos[i]==time_choice_pos and time_choice_method=="default")*time_choice_outdim, layers_hidden[feature_insert_pos[i]+1],
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
                        self.layers[time_choice_pos] = FastKANLayer(
                            layers_hidden[time_choice_pos] + time_choice_outdim, layers_hidden[time_choice_pos+1],
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

        elif self.moe_enabled:
            num_model_layers = len(layers_hidden) - 1
            if not isinstance(expert_pos, int) or not 0 <= expert_pos < num_model_layers:
                raise ValueError(
                    "expert_pos must be a valid layer index in "
                    "[0, len(layers_hidden) - 2]"
                )

            insert_dims = feature_num if isinstance(feature_num, list) else []
            if len(feature_insert_pos) != len(insert_dims):
                raise ValueError(
                    "feature_insert_pos and feature_insert must have equal lengths"
                )

            # Features originally scheduled at or after the expert boundary are
            # all inserted once, immediately before the first expert layer.
            self.moe_feature_insert_map = {}
            for feature_idx, original_pos in enumerate(feature_insert_pos):
                original_pos = int(original_pos)
                if not 0 <= original_pos < num_model_layers:
                    raise ValueError(
                        f"Invalid feature_insert_pos {original_pos} for "
                        f"{num_model_layers} layers"
                    )
                effective_pos = (
                    original_pos if original_pos < expert_pos else expert_pos
                )
                self.moe_feature_insert_map.setdefault(effective_pos, []).append(
                    feature_idx
                )

            insert_dim_by_pos = {
                pos: sum(int(insert_dims[idx]) for idx in feature_indices)
                for pos, feature_indices in self.moe_feature_insert_map.items()
            }

            def build_fastkan_layer(layer_idx):
                return FastKANLayer(
                    layers_hidden[layer_idx] + insert_dim_by_pos.get(layer_idx, 0),
                    layers_hidden[layer_idx + 1],
                    grid_min=grid_min,
                    grid_max=grid_max,
                    num_grids=self.num_grids[layer_idx],
                    use_base_update=use_base_update,
                    base_activation=base_activation,
                    spline_weight_init_scale=spline_weight_init_scale,
                    denominator=self.denominator[layer_idx],
                    dropout_rate=dropout_rate,
                    use_layernorm=self.use_layernorm,
                    radial_type=radial_type,
                    res_net=res_net,
                    device=device,
                    infer=infer,
                )

            # Keep a complete original/base path for checkpoint compatibility
            # and a stable residual prediction.
            self.layers = nn.ModuleList([
                build_fastkan_layer(i) for i in range(num_model_layers)
            ])

            self.expert_layers = nn.ModuleList([
                ParallelExpertFastKANLayer(
                    num_experts=num_experts,
                    input_dim=(
                        layers_hidden[i] + insert_dim_by_pos.get(i, 0)
                    ),
                    output_dim=layers_hidden[i + 1],
                    grid_min=grid_min,
                    grid_max=grid_max,
                    num_grids=self.num_grids[i],
                    use_base_update=use_base_update,
                    base_activation=base_activation,
                    spline_weight_init_scale=spline_weight_init_scale,
                    denominator=self.denominator[i],
                    dropout_rate=dropout_rate,
                    use_layernorm=self.use_layernorm,
                    radial_type=radial_type,
                    res_net=res_net,
                    device=device,
                    infer=infer,
                )
                for i in range(expert_pos, num_model_layers)
            ])

        if self.res_net == "attn_resnet":
            if attn_res_pos is None:
                attn_res_pos = list(range(len(self.layers)-2))
            res_outdim = self.layers[attn_res_pos[-1]].output_dim
            self.attn_res_pos = np.array(attn_res_pos[:-1])
            self.attn_target = attn_res_pos[-1]
            if infer:
                self.attn_res_layers = nn.ModuleList([
                    skip_init(nn.Linear, self.layers[attn_res_pos[i]].output_dim, res_outdim, device=device)
                for i in range(len(attn_res_pos)-1)])
            else:
                self.attn_res_layers = nn.ModuleList([
                    nn.Linear(self.layers[attn_res_pos[i]].output_dim, res_outdim, device=device)
                for i in range(len(attn_res_pos)-1)])
            self.rmsnorms = nn.ModuleList([RMSNorm(res_outdim, device=device) for _ in range(len(attn_res_pos)-1)])
            if infer:
                self.Query = nn.Parameter(torch.empty(res_outdim, device=device))
            else:
                self.Query = nn.Parameter(torch.randn(res_outdim, device=device))

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

    def _forward_moe(self, x):
        market_feature = x[:, self.time_choice_feature]
        router_weights = self.time_choice_block(market_feature)

        # Only the masks referenced by feature_insert_pos are inserted into the
        # model.  The final mask in self.feature_insert belongs to the router.
        insert_features = [x[:, self.feature_insert[feature_idx]] for feature_idx in range(len(self.feature_insert_pos))]
        remove_mask = self.feature_insert.any(axis=0)
        x = x[:, ~remove_mask]

        if self.stem_component:
            x = self.stem_component.forward(x)

        def insert_at(layer_input, layer_idx):
            feature_indices = self.moe_feature_insert_map.get(layer_idx, [])
            if feature_indices:
                layer_input = torch.cat([layer_input] + [insert_features[idx] for idx in feature_indices], dim=-1)
            return layer_input

        # Shared trunk.
        for layer_idx in range(self.expert_pos):
            x = insert_at(x, layer_idx)
            if self.use_tanh:
                x = self.tanh_norm * torch.tanh(x / self.tanh_denom)
            x = self.layers[layer_idx](x)

        # All features assigned to expert layers have been relocated here.
        x = insert_at(x, self.expert_pos)
        base_output = x
        expert_output = x.unsqueeze(-2).expand(*x.shape[:-1], self.num_experts, x.shape[-1])

        for expert_offset, layer_idx in enumerate(range(self.expert_pos, len(self.layers))):
            if self.use_tanh:
                base_output = self.tanh_norm * torch.tanh(base_output / self.tanh_denom)
                expert_output = self.tanh_norm * torch.tanh(expert_output / self.tanh_denom)
            base_output = self.layers[layer_idx](base_output)
            expert_output = self.expert_layers[expert_offset](expert_output)

        moe_output = torch.einsum("...e,...eo->...o", router_weights, expert_output)
        # Detached weights are useful for diagnosing expert collapse without
        # retaining the autograd graph between training iterations.
        self.expert_output = expert_output.detach()
        self.last_router_weights = router_weights.detach()
        return base_output + self.moe_residual_scale * moe_output

    def forward(self, x):
        if self.moe_enabled:
            return self._forward_moe(x)

        if (
            self.feature_insert_pos is not None
            and self.feature_insert is not None
            and np.asarray(self.feature_insert).size > 0
        ):
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
                if self.res_net == "attn_resnet":
                    keys = []
                    values = []
                if self.time_choice_pos is not None or self.time_choice_method == "barra_estimator":
                    time_choice_feature = x[:, self.time_choice_feature]
                    if self.time_choice_method == "barra_estimator":
                        barra_section = x[..., -self.barra_dim:]
                    time_choice_result = self.time_choice_block(time_choice_feature)
                feature_insert = []
                for features_indices in self.feature_insert:
                    feature_insert.append(x[:, features_indices])
                x = x[:, ~(self.feature_insert.any(axis=0))]
                if self.stem_component:
                    x = self.stem_component.forward(x)
                for i in range(len(self.layers)):
                    if self.time_choice_pos is not None and i == self.time_choice_pos:
                        if self.time_choice_method == "linear_modulation":
                            lm_res = time_choice_result
                            x = x * lm_res[:,:self.lm_dim] + lm_res[:,self.lm_dim:]
                        elif self.time_choice_method == "barra_estimator":
                            x = x
                        else:
                            x = torch.cat([x, time_choice_result], dim=1)
                    if i in self.feature_insert_pos:
                        idx = np.where(self.feature_insert_pos==i)[0][0]
                        x = torch.cat([x, feature_insert[idx]], dim=1)
                    if self.use_tanh:
                        x = self.tanh_norm * torch.tanh(x/self.tanh_denom)
                    x = self.layers[i](x)
                    if self.res_net == "attn_resnet":
                        if i in self.attn_res_pos:
                            idx = np.where(self.attn_res_pos==i)[0][0]
                            v = self.attn_res_layers[idx](x)
                            k = self.rmsnorms[idx](v)
                            keys.append(k.unsqueeze(0))
                            values.append(v.unsqueeze(0))
                        elif i == self.attn_target:
                            keys = torch.cat(keys, dim=0)
                            values = torch.cat(values, dim=0)
                            att = torch.exp(torch.einsum("f,lbf->lb", self.Query, keys))
                            att = att / att.sum(dim=0, keepdim=True)
                            attn_output = torch.einsum("lb, lbf->lbf", att, values).sum(dim=0)
                            if self.use_tanh:
                                attn_output = self.tanh_norm * torch.tanh(attn_output/self.tanh_denom)
                            x = x + attn_output
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
