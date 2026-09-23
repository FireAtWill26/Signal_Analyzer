from .fastkan import *
from .fastkan1 import Injector
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import skip_init
from copy import deepcopy

def BuildLayer(in_dim, out_dim, grid_min, grid_max, num_grids, use_base_update, base_activation, spline_weight_init_scale, denominator, dropout_rate, use_layernorm, radial_type, mode, device="cpu", infer=False):
    if mode == "KAF":
        layer = FastKAFLayer(in_dim, out_dim, num_grids, use_layernorm, dropout_rate, base_activation, spline_weight_init_scale, device=device, infer=infer)
    elif mode == "Cheby":
        layer = ChebyKANLayer(in_dim, out_dim, num_grids, spline_weight_init_scale, dropout_rate, device=device, infer=infer)
    else:
        layer = FastKANLayer(in_dim, out_dim, grid_min, grid_max, num_grids, use_base_update, use_layernorm, base_activation, spline_weight_init_scale, denominator, dropout_rate, radial_type, res_net=True, device=device, infer=infer)
    return layer

# class AttentionWithFastKANTransformKVCache(nn.Module):
    
#     def __init__(
#         self,
#         q_dim: int,
#         k_dim: int,
#         v_dim: int,
#         head_dim: int,
#         num_heads: int,
#         grid_min = -2,
#         grid_max = 2,
#         num_grids = 16,
#         base_activation = F.gelu,
#         spline_weight_init_scale = 1e-3,
#         use_layernorm = True,
#         denominator = 0.33,
#         dropout_rate = 0.2,
#         radial_type = "Gaussian",
#         use_tanh = False,
#         mode = "default",
#         gating: bool = True,
#     ):
#         super(AttentionWithFastKANTransformKVCache, self).__init__()

#         self.num_heads = num_heads
#         self.use_tanh = use_tanh
#         total_dim = head_dim * self.num_heads
#         self.gating = gating
#         self.linear_q = BuildLayer(q_dim, total_dim, grid_max, grid_min, num_grids, True, base_activation, spline_weight_init_scale, denominator, dropout_rate, use_layernorm, radial_type, mode)
#         self.linear_k = BuildLayer(k_dim, total_dim, grid_max, grid_min, num_grids, True, base_activation, spline_weight_init_scale, denominator, dropout_rate, use_layernorm, radial_type, mode)
#         self.linear_v = BuildLayer(v_dim, total_dim, grid_max, grid_min, num_grids, True, base_activation, spline_weight_init_scale, denominator, dropout_rate, use_layernorm, radial_type, mode)
#         self.linear_o = BuildLayer(total_dim, q_dim, grid_max, grid_min, num_grids, True, base_activation, spline_weight_init_scale, denominator, dropout_rate, use_layernorm, radial_type, mode)
#         self.linear_g = None
#         if self.gating:
#             self.linear_g = BuildLayer(q_dim, total_dim, grid_max, grid_min, num_grids, True, base_activation, spline_weight_init_scale, denominator, dropout_rate, use_layernorm, radial_type, mode)
#         # precompute the 1/sqrt(head_dim)
#         self.norm = head_dim**-0.5

#     def forward(
#         self,
#         q: torch.Tensor,
#         k: torch.Tensor,
#         v: torch.Tensor,
#         bias: torch.Tensor = None,      # additive attention bias
#     ) -> torch.Tensor:
#         k_cache = []
#         v_cache = []
#         o_cache = []
#         if self.use_tanh:
#             q = torch.tanh(q)
#             k = torch.tanh(k)
#             v = torch.tanh(v)
#         # here we take q = k = v = x, where x.shape = batch_size * seq_len * input_dim
#         # from here, we denote b = batch_size, s = seq_len, n = num_heads, h = head_dim
#         batch_size, seq_len, _ = q.shape
#         for i in range(seq_len):
#             # wq.shape = b, 1, 1, n, h
#             wq = self.linear_q(q[:,i:i+1,:]).view(*q.shape[:-1], 1, self.num_heads, -1) * self.norm
#             # wk.shape = b, 1, 1, n, h
#             wk = self.linear_k(k[:,i:i+1,:]).view(*k.shape[:-2], 1, k.shape[-2], self.num_heads, -1)
#             k_cache.append(wk.detach())
#             # wk.shape = b, 1, s, n, h
#             wk = torch.cat(k_cache, dim=2)
#             # (wq * wk).shape = b, 1, s, n, h
#             # att.shape = b, 1, s, n
#             att = (wq * wk).sum(-1).softmax(-2)
#             del wq, wk
#             if bias is not None:
#                 att = att + bias[..., None]
#             # wv.shape = b, 1, 1, n, h
#             wv = self.linear_v(v[:,i:i+1,:]).view(*v.shape[:-2],1, v.shape[-2], self.num_heads, -1)
#             v_cache.append(wv.detach())
#             # wv.shape = b, 1, s, n, h
#             wv = torch.cat(wv, dim=2)
#             # (att[...,None] * wv).shape = b, 1, s, n, h
#             # o.shape = b, 1, n, h
#             o = (att[...,None] * wv).sum(-3)
#             del att, wv
#             # o.shape = b, 1, (nh)
#             o = o.view(*o.shape[:-2], -1)
#             o_cache.append(o.detach())

#         o = torch.cat(o_cache, dim=1)

#         if self.linear_g is not None:
#             # gating, use raw query input
#             g = self.linear_g(q)
#             o = torch.sigmoid(g) * o

#         # merge heads
#         o = self.linear_o(o)
#         return o


class AttentionWithFastKANTransform(nn.Module):
    
    def __init__(
        self,
        q_dim: int,
        k_dim: int,
        v_dim: int,
        head_dim: int,
        num_heads: int,
        grid_min = -2,
        grid_max = 2,
        num_grids = 16,
        base_activation = F.gelu,
        spline_weight_init_scale = 1e-3,
        use_layernorm = True,
        denominator = 0.33,
        dropout_rate = 0.2,
        radial_type = "Gaussian",
        use_tanh = False,
        mode = "default",
        gating: bool = True,
        mask: bool = False,
        device = "cpu",
        infer = False,
    ):
        super(AttentionWithFastKANTransform, self).__init__()

        self.num_heads = num_heads
        # self.use_tanh = use_tanh
        total_dim = head_dim * self.num_heads
        self.gating = gating
        self.mask = mask
        self.linear_q = BuildLayer(q_dim, total_dim, grid_max, grid_min, num_grids, True, base_activation, spline_weight_init_scale, denominator, dropout_rate=0, use_layernorm=use_layernorm, radial_type=radial_type, mode=mode, device=device, infer=infer)
        self.linear_k = BuildLayer(k_dim, total_dim, grid_max, grid_min, num_grids, True, base_activation, spline_weight_init_scale, denominator, dropout_rate=0, use_layernorm=use_layernorm, radial_type=radial_type, mode=mode, device=device, infer=infer)
        self.linear_v = BuildLayer(v_dim, total_dim, grid_max, grid_min, num_grids, True, base_activation, spline_weight_init_scale, denominator, dropout_rate=0, use_layernorm=use_layernorm, radial_type=radial_type, mode=mode, device=device, infer=infer)
        self.linear_o = BuildLayer(total_dim, q_dim, grid_max, grid_min, num_grids, True, base_activation, spline_weight_init_scale, denominator, dropout_rate=0, use_layernorm=use_layernorm, radial_type=radial_type, mode=mode, device=device, infer=infer)
        self.linear_g = None
        if self.gating:
            self.linear_g = BuildLayer(q_dim, total_dim, grid_max, grid_min, num_grids, True, base_activation, spline_weight_init_scale, denominator, dropout_rate=0, use_layernorm=use_layernorm, radial_type=radial_type, mode=mode, device=device, infer=infer)
        # precompute the 1/sqrt(head_dim)
        self.norm = head_dim**-0.5
        if dropout_rate != 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        bias: torch.Tensor = None,      # additive attention bias
        infer: bool = False
    ) -> torch.Tensor:
        if self.mask:
            mask = torch.ones(q.shape[0], 1, q.shape[1], q.shape[1], device=q.device)
            mask = mask.tril_()

        if infer:

            # here we have x.shape = batch_size * seq_len * input_dim
            # from here, we denote b = batch_size, s = seq_len, n = num_heads, h = head_dim
            # q =self.linear_q(x), k = self.linear_k(x), v = self.linear_v(x)
            # q.shape = k.shape = v.shape = [b, s, n * h]
            # wq.shape = [b, h, 1, n]

            wq = self.linear_q(q[:,-1,:]).view(q.shape[0], 1, self.num_heads, -1).transpose(1,2) * self.norm
            # wk.shape = [b, h, n, s]
            wk = self.linear_k(k).view(*k.shape[:-1], self.num_heads, -1).transpose(1,2).transpose(-2,-1)
            # score.shape = [b, h, 1, s]
            score = torch.matmul(wq, wk)

            # if self.mask:
            #     score = score.masked_fill(mask == 0, -1e9)

            # att.shape = [b, h, 1, s]
            att = score.softmax(-1)
            if self.dropout is not None:
                att = self.dropout(att)

            # del wq, wk, score
            if bias is not None:
                att = att + bias[..., None]

            # wv.shape = [b, h, s, n]
            wv = self.linear_v(v).view(*v.shape[:-1], self.num_heads, -1).transpose(1,2)

            # o.shape = [b, h, 1, n]
            o = torch.matmul(att, wv)
            # print(o.shape)
            # del att, wv

            # o.shape = [b, 1, h * n]
            o = o.transpose(1,2).contiguous()
            o = o.view(*o.shape[:-2], -1)

            if self.linear_g is not None:
                # gating, use raw query input
                g = self.linear_g(q)
                # print(g.shape, o.shape)
                o = torch.sigmoid(g) * o

            # merge heads
            o = self.linear_o(o)
            return o

        # if infer:
        #     # here we have x.shape = batch_size * seq_len * input_dim
        #     # from here, we denote b = batch_size, s = seq_len, n = num_heads, h = head_dim
        #     # q =self.linear_q(x), k = self.linear_k(x), v = self.linear_v(x)
        #     # q.shape = k.shape = v.shape = [b, s, n * h]
        #     # wq.shape = [b, h, 1, n]
        #     wq = self.linear_q(q[:,-1,:]).view(*q.shape[:-2], 1, self.num_heads, -1).transpose(1,2) * self.norm
        #     # wk.shape = [b, h, n, s]
        #     wk = self.linear_k(k).view(*k.shape[:-1], self.num_heads, -1).transpose(1,2).transpose(-2,-1)
        #     # score.shape = [b, h, 1, s]
        #     score = torch.matmul(wq, wk)
        #     # att.shape = [b, h, 1, s]
        #     att = score.softmax(-1)
        #     del wq, wk, score
        #     if bias is not None:
        #         att = att + bias[..., None]
        #     # wv.shape = [b, h, s, n]
        #     wv = self.linear_v(v).view(*v.shape[:-1], self.num_heads, -1).transpose(1,2)
        #     # o.shape = [b, h, 1, n]
        #     o = torch.matmul(att, wv)
        #     del att, wv
            
        #     # o.shape = [b, 1, h * n]
        #     o = o.transpose(1,2).contiguous()
        #     o = o.view(*o.shape[:-2], -1)
 
        #     if self.linear_g is not None:
        #         # gating, use raw query input
        #         g = self.linear_g(q)
        #         o = torch.sigmoid(g) * o

        #     # merge heads
        #     o = self.linear_o(o)
        #     return o
            
        # here we have x.shape = batch_size * seq_len * input_dim
        # from here, we denote b = batch_size, s = seq_len, n = num_heads, h = head_dim
        # q =self.linear_q(x), k = self.linear_k(x), v = self.linear_v(x)
        # q.shape = k.shape = v.shape = [b, s, n * h]
        # wq.shape = [b, h, s, n]
        wq = self.linear_q(q).view(*q.shape[:-1], self.num_heads, -1).transpose(1,2) * self.norm
        # wk.shape = [b, h, n, s]
        wk = self.linear_k(k).view(*k.shape[:-1], self.num_heads, -1).transpose(1,2).transpose(-2,-1)
        # score.shape = [b, h, s, s]
        score = torch.matmul(wq, wk)

        if self.mask:
            score = score.masked_fill(mask == 0, -1e9)

        # att.shape = [b, h, s, s]
        att = score.softmax(-1)
        if self.dropout is not None:
            att = self.dropout(att)

        # del wq, wk, score
        if bias is not None:
            att = att + bias[..., None]

        # wv.shape = [b, h, s, n]
        wv = self.linear_v(v).view(*v.shape[:-1], self.num_heads, -1).transpose(1,2)

        # o.shape = [b, h, s, n]
        o = torch.matmul(att, wv)
        # print(o.shape)
        # del att, wv

        # o.shape = [b, s, h * n]
        o = o.transpose(1,2).contiguous()
        o = o.view(*o.shape[:-2], -1)

        if self.linear_g is not None:
            # gating, use raw query input
            g = self.linear_g(q)
            # print(g.shape, o.shape)
            o = torch.sigmoid(g) * o

        # merge heads
        o = self.linear_o(o)
        return o

class KDA(nn.Module):
    """Kimi Delta Attention with FastKAN projection layers.

    This is a pure PyTorch recurrent implementation of KDA.  It follows the
    fine-grained (per key dimension) decay and delta-rule state update used by
    Kimi Linear, while using ``BuildLayer`` for every learned projection.  The
    recurrent implementation is intentionally kept dependency-free and is a
    good fit for the relatively short financial sequences used by this model.

    KDA is causal by construction.  ``bias`` may therefore only be a 2-D
    padding mask of shape ``[batch, seq_len]``; arbitrary pairwise attention
    biases cannot be represented by the recurrent state transition.
    """

    def __init__(
        self,
        q_dim: int,
        k_dim: int,
        v_dim: int,
        head_dim: int,
        num_heads: int,
        grid_min = -2,
        grid_max = 2,
        num_grids = 16,
        base_activation = F.gelu,
        spline_weight_init_scale = 1e-3,
        use_layernorm = True,
        denominator = 0.33,
        dropout_rate = 0.2,
        radial_type = "Gaussian",
        use_tanh = False,
        mode = "default",
        gating: bool = True,
        mask: bool = False,
        device = "cpu",
        infer = False,
        conv_size: int = 4,
        rms_norm_eps: float = 1e-5,
    ):
        super().__init__()
        if head_dim <= 0 or num_heads <= 0:
            raise ValueError("head_dim and num_heads must both be positive")
        if conv_size <= 0:
            raise ValueError("conv_size must be positive")

        self.head_dim = head_dim
        self.num_heads = num_heads
        self.total_dim = head_dim * num_heads
        self.gating = gating
        self.mask = mask
        self.conv_size = conv_size
        self.rms_norm_eps = rms_norm_eps
        self.norm = head_dim ** -0.5

        def build(in_dim, out_dim, layernorm=use_layernorm):
            return BuildLayer(
                in_dim=in_dim,
                out_dim=out_dim,
                grid_min=grid_min,
                grid_max=grid_max,
                num_grids=num_grids,
                use_base_update=True,
                base_activation=base_activation,
                spline_weight_init_scale=spline_weight_init_scale,
                denominator=denominator,
                dropout_rate=0,
                use_layernorm=layernorm,
                radial_type=radial_type,
                mode=mode,
                device=device,
                infer=infer,
            )

        # Q/K/V projections.  The following depth-wise convolutions are the
        # short causal convolutions from KDA; all dense projections are FastKAN.
        self.linear_q = build(q_dim, self.total_dim)
        self.linear_k = build(k_dim, self.total_dim)
        self.linear_v = build(v_dim, self.total_dim)
        self.q_conv1d = nn.Conv1d(
            self.total_dim, self.total_dim, conv_size,
            groups=self.total_dim, padding=conv_size - 1, device=device,
        )
        self.k_conv1d = nn.Conv1d(
            self.total_dim, self.total_dim, conv_size,
            groups=self.total_dim, padding=conv_size - 1, device=device,
        )
        self.v_conv1d = nn.Conv1d(
            self.total_dim, self.total_dim, conv_size,
            groups=self.total_dim, padding=conv_size - 1, device=device,
        )

        # Fine-grained forget gate: q_dim -> head_dim -> num_heads * head_dim.
        self.linear_f_a = build(q_dim, head_dim)
        self.linear_f_b = build(head_dim, self.total_dim)

        # Per-head delta-rule write rate beta.
        self.linear_beta = build(q_dim, num_heads)

        # KDA's output gate and final output projection.
        if gating:
            self.linear_g_a = build(q_dim, head_dim)
            self.linear_g_b = build(head_dim, self.total_dim)
        else:
            self.linear_g_a = None
            self.linear_g_b = None
        self.linear_o = build(self.total_dim, q_dim)

        # A_log controls the decay rate of each head.  dt_bias is initialized
        # so softplus(dt_bias) is log-uniform in [1e-3, 1e-1], preventing the
        # recurrent memory from being erased immediately at initialization.
        self.A_log = nn.Parameter(
            torch.log(torch.empty(num_heads, device=device).uniform_(1.0, 16.0))
        )
        dt = torch.exp(
            torch.empty(self.total_dim, device=device).uniform_(
                math.log(1e-3), math.log(1e-1)
            )
        )
        inverse_softplus_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inverse_softplus_dt)

        # Per-head RMSNorm scale, matching KDA's RMSNorm-gated output stage.
        self.o_norm_weight = nn.Parameter(
            torch.ones(head_dim, device=device)
        )
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

    def _causal_short_conv(self, x, conv):
        """Apply a depth-wise causal convolution to ``[B, T, D]`` input."""
        seq_len = x.shape[1]
        x = conv(x.transpose(1, 2))[..., :seq_len].transpose(1, 2)
        return F.silu(x)

    def _padding_mask(self, bias, reference):
        if bias is None:
            return None
        if bias.ndim != 2 or bias.shape != reference.shape[:2]:
            raise ValueError(
                "KDA only supports a padding mask with shape "
                f"[batch, seq_len]; got {tuple(bias.shape)}"
            )
        return bias.to(device=reference.device, dtype=torch.bool)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        bias: torch.Tensor = None,
        infer: bool = False,
    ) -> torch.Tensor:
        del infer  # Recurrent and full-sequence paths are numerically identical.

        if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
            raise ValueError("q, k and v must all have shape [batch, seq_len, dim]")
        if q.shape[:2] != k.shape[:2] or q.shape[:2] != v.shape[:2]:
            raise ValueError("q, k and v must share batch and sequence dimensions")

        padding_mask = self._padding_mask(bias, q)

        q_proj = self.linear_q(q)
        k_proj = self.linear_k(k)
        v_proj = self.linear_v(v)
        if padding_mask is not None:
            projection_mask = padding_mask.unsqueeze(-1).to(q_proj.dtype)
            q_proj = q_proj * projection_mask
            k_proj = k_proj * projection_mask
            v_proj = v_proj * projection_mask

        q_proj = self._causal_short_conv(q_proj, self.q_conv1d)
        k_proj = self._causal_short_conv(k_proj, self.k_conv1d)
        v_proj = self._causal_short_conv(v_proj, self.v_conv1d)

        batch_size, seq_len = q.shape[:2]
        if seq_len == 0:
            raise ValueError("KDA does not support an empty sequence")
        q_proj = q_proj.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k_proj = k_proj.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v_proj = v_proj.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # The official kernel performs Q/K L2 normalization internally.
        q_proj = F.normalize(q_proj.float(), p=2.0, dim=-1, eps=1e-6)
        k_proj = F.normalize(k_proj.float(), p=2.0, dim=-1, eps=1e-6)
        v_proj = v_proj.float()

        forget_logits = self.linear_f_b(self.linear_f_a(q)).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        )
        decay_log = -self.A_log.view(1, 1, self.num_heads, 1).exp() * F.softplus(
            forget_logits.float()
            + self.dt_bias.view(1, 1, self.num_heads, self.head_dim)
        )
        beta = torch.sigmoid(self.linear_beta(q).float())

        # S has a fixed [key_dim, value_dim] matrix per head.  Updating it in
        # float32 is important for stable products over long sequences.
        state = torch.zeros(
            batch_size,
            self.num_heads,
            self.head_dim,
            self.head_dim,
            dtype=torch.float32,
            device=q.device,
        )
        outputs = []
        for i in range(seq_len):
            q_i = q_proj[:, i]
            k_i = k_proj[:, i]
            v_i = v_proj[:, i]

            decayed_state = state * decay_log[:, i].exp().unsqueeze(-1)
            old_value = torch.einsum("bhk,bhkv->bhv", k_i, decayed_state)
            value_error = v_i - old_value
            updated_state = decayed_state + torch.einsum(
                "bhk,bhv->bhkv",
                beta[:, i].unsqueeze(-1) * k_i,
                value_error,
            )

            if padding_mask is not None:
                valid = padding_mask[:, i].view(batch_size, 1, 1, 1)
                state = torch.where(valid, updated_state, state)
            else:
                state = updated_state

            o_i = torch.einsum("bhk,bhkv->bhv", q_i * self.norm, state)
            if padding_mask is not None:
                o_i = o_i * padding_mask[:, i].view(batch_size, 1, 1)
            outputs.append(o_i)

        output = torch.stack(outputs, dim=1)

        # Head-wise RMSNorm followed by KDA's sigmoid output gate.
        output = output * torch.rsqrt(
            output.square().mean(dim=-1, keepdim=True) + self.rms_norm_eps
        )
        output = output * self.o_norm_weight.view(1, 1, 1, self.head_dim)
        if self.gating:
            output_gate = self.linear_g_b(self.linear_g_a(q)).view(
                batch_size, seq_len, self.num_heads, self.head_dim
            )
            output = output * torch.sigmoid(output_gate.float())

        output = output.to(q.dtype).reshape(batch_size, seq_len, self.total_dim)
        if self.dropout is not None:
            output = self.dropout(output)
        output = self.linear_o(output)
        if padding_mask is not None:
            output = output * padding_mask.unsqueeze(-1).to(output.dtype)
        return output

class KANFeedForward(nn.Module):
    def __init__(
        self,
        in_dim: int,
        expand_dim: int,
        grid_min = -2,
        grid_max = 2,
        num_grids = 16,
        base_activation = F.gelu,
        spline_weight_init_scale = 1e-3,
        use_layernorm = True,
        denominator = 0.33,
        dropout_rate = 0.2,
        radial_type = "Gaussian",
        use_tanh = False,
        mode = "default",
        device = "cpu",
        infer = False,
    ):
        super().__init__()
        self.LinearLayer1 = BuildLayer(in_dim, expand_dim, grid_max, grid_min, num_grids, True, base_activation, spline_weight_init_scale, denominator, dropout_rate, use_layernorm, radial_type, mode, device=device, infer=infer)
        self.LinearLayer2 = BuildLayer(expand_dim, in_dim, grid_max, grid_min, num_grids, True, base_activation, spline_weight_init_scale, denominator, dropout_rate, use_layernorm, radial_type, mode, device=device, infer=infer)
        self.use_tanh = use_tanh
        self.drop = nn.Dropout(dropout_rate)

    def forward(self, x):
        # if self.use_tanh:
        #     x = torch.tanh(x)
        x = self.LinearLayer1(x)
        x = self.drop(x)
        # if self.use_tanh:
        #     x = torch.tanh(x)
        x = self.LinearLayer2(x)
        return x


class KANsformerLayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        expand_dim: int,
        head_dim: int,
        num_heads: int,
        grid_min = -2,
        grid_max = 2,
        num_grids = 16,
        base_activation = F.gelu,
        spline_weight_init_scale = 1e-3,
        use_layernorm = True,
        denominator = 0.33,
        dropout_rate = 0.2,
        radial_type = "Gaussian",
        use_tanh = False,
        mode = "default",
        gating: bool = True,
        mask: bool = False,
        device = "cpu",
        infer = False, ):
        super().__init__()
        
        self.use_tanh = use_tanh
        self.use_layernorm = use_layernorm
        self.attn = AttentionWithFastKANTransform(in_dim, in_dim, in_dim, head_dim, num_heads, grid_min, grid_max, num_grids, base_activation, spline_weight_init_scale, use_layernorm, denominator, dropout_rate, radial_type, use_tanh, mode, gating, mask, device=device, infer=infer)
        self.feed_forward = KANFeedForward(in_dim, expand_dim, grid_min, grid_max, num_grids, base_activation, spline_weight_init_scale, use_layernorm, denominator, dropout_rate, radial_type, use_tanh, mode, device=device, infer=infer)
        if use_layernorm:
            if infer:
                self.norm1 = skip_init(nn.LayerNorm, in_dim, device=device)
                self.norm2 = skip_init(nn.LayerNorm, in_dim, device=device)
            else:
                self.norm1 = nn.LayerNorm(in_dim, device=device)
                self.norm2 = nn.LayerNorm(in_dim, device=device)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x, infer=False, bias = None):
        # if self.use_tanh:
        #     x = torch.tanh(x)
        attn_output = self.attn(x,x,x,bias,infer)
        if self.use_layernorm:
            x = self.norm1(x + self.dropout1(attn_output))
        else:
            x = x + self.dropout1(attn_output)
        ffn_output = self.feed_forward(x)
        if self.use_layernorm:
            x = self.norm2(x + self.dropout2(ffn_output))
        else:
            x = x + self.dropout2(ffn_output)
        # del ffn_output
        return x

@MODEL_REGISTRY.register('KANsformer')
class KANsformer(nn.Module):
    def __init__(
        self,
        num_layers: int,
        input_dim: int,
        embedding_dim,
        expand_dim: int,
        head_dim: int,
        num_heads: int,
        grid_min = -2,
        grid_max = 2,
        num_grids = 16,
        num_grids_embed = 16,
        num_grids_decode = 16,
        base_activation = F.gelu,
        spline_weight_init_scale = 1e-3,
        denominator = 0.33,
        use_layernorm = True,
        dropout_rate = 0.2,
        radial_type = "Gaussian",
        use_tanh = False,
        tanh_norm = 2,
        tanh_denom = 1,
        mode = "default",
        mask: bool = False,
        seq_len = 20,
        gating: bool = True,
        residual: bool = False,
        feature_insert = None,
        time_choice_feature = None,
        barra_dim = None,
        trident = False,
        trident_merge_method = "concat",
        trident_heads = None,
        device = "cpu",
        infer = False, ):
        super().__init__()

        # self.use_tanh = use_tanh
        self.infer = infer
        self.residual = residual
        self.use_tanh = use_tanh
        self.tanh_norm = tanh_norm
        self.tanh_denom = tanh_denom

        self.feature_insert = feature_insert

        self.time_choice_feature = time_choice_feature

        added_input_dim = added_output_dim =  0

        if feature_insert is not None:
            added_input_dim += feature_insert.sum()
            added_output_dim += feature_insert.sum()
        if time_choice_feature is not None:
            added_input_dim += time_choice_feature.sum()

        if trident:
            if time_choice_feature is not None or feature_insert is not None:
                if time_choice_feature is not None:
                    to_remove = ~time_choice_feature
                else:
                    to_remove = np.ones_like(insert_feature[0], dtype=bool)
                for insert_feature in feature_insert:
                    to_remove &= ~insert_feature
            else:
                to_remove = np.ones_like(trident_heads[0], dtype=bool)

        else:
            input_dim -= added_input_dim

        if time_choice_feature is not None:
            self.barra_dim=barra_dim        
            self.time_choice_block = Injector(
                time_choice_feature.sum(),
                self.barra_dim,
                grid_min=grid_min,
                grid_max=grid_max,
                num_grids=8,
                use_base_update=True,
                base_activation=base_activation,
                spline_weight_init_scale=spline_weight_init_scale,
                denominator=0.33,
                use_layernorm=False,
                dropout_rate=dropout_rate,
                radial_type=radial_type,
                res_net=True,
                use_tanh=use_tanh,
                tanh_norm=tanh_norm,
                tanh_denom=tanh_denom,
                device=device,
                infer=infer,
                )

        self.trident= trident
        self.trident_merge_method = trident_merge_method
        if trident:
            trident_len = len(embedding_dim) - 1
            if trident_merge_method == "EW_ensemble":
                trident_len += 1
            self.spear_heads = nn.ModuleList()
            next_in_dim=0
            self.embedding_dim = embedding_dim[-1]
            for head in trident_heads:
                trident_head_dim = (head & to_remove).sum()
                trident_dim = [math.ceil(embedding_dim[i] * trident_head_dim) for i in range(len(embedding_dim)-1)]
                # print(trident_dim)
                if trident_merge_method == "EW_ensemble":
                    trident_dim += [self.embedding_dim]
                else:
                    next_in_dim += trident_dim[-1]
                self.spear_heads.append(nn.ModuleList([FastKANLayer(
                            trident_dim[j], trident_dim[j+1],
                            grid_min=grid_min,
                            grid_max=grid_max,
                            num_grids=num_grids_embed,
                            use_base_update=True,
                            base_activation=base_activation,
                            spline_weight_init_scale=spline_weight_init_scale,
                            denominator=denominator,
                            dropout_rate=dropout_rate,
                            use_layernorm=False,
                            radial_type=radial_type,
                            res_net=True,
                            device=device,
                            infer=infer,
                        ) for j in range(trident_len-1)]))
            if trident_merge_method == "concat":
                self.embedding = FastKANLayer(
                            next_in_dim, self.embedding_dim,
                            grid_min=grid_min,
                            grid_max=grid_max,
                            num_grids=num_grids_embed,
                            use_base_update=True,
                            base_activation=base_activation,
                            spline_weight_init_scale=spline_weight_init_scale,
                            denominator=denominator,
                            dropout_rate=dropout_rate,
                            use_layernorm=False,
                            radial_type=radial_type,
                            res_net=True,
                            device=device,
                            infer=infer,
                        )            
            self.trident_heads = [trident_head & to_remove for trident_head in trident_heads]
        else:
            if type(embedding_dim) == int:
                self.embedding_dim = embedding_dim
                self.Embedding = nn.ModuleList([BuildLayer(input_dim, embedding_dim, grid_max, grid_min, num_grids, True, base_activation, spline_weight_init_scale, denominator, 0, False, radial_type, mode, device=device, infer=infer)])
            if isinstance(embedding_dim, list):
                self.embedding_dim = embedding_dim[-1]
                if len(embedding_dim) > 1:
                    embedding_layers = []
                    for i in range(len(embedding_dim) - 1):
                        embedding_layers.append(math.ceil(embedding_dim[i] * input_dim))
                    embedding_layers.append(self.embedding_dim)
                    # print(embedding_layers)
                    self.Embedding = nn.ModuleList([BuildLayer(in_dim, out_dim, grid_max, grid_min, num_grids_embed, True, base_activation, spline_weight_init_scale, denominator, 0, False, radial_type, mode, device=device, infer=infer) for in_dim, out_dim in zip(embedding_layers[:-1], embedding_layers[1:])])
                else:
                    self.Embedding = nn.ModuleList([BuildLayer(input_dim, self.embedding_dim, grid_max, grid_min, num_grids_embed, True, base_activation, spline_weight_init_scale, denominator, 0, False, radial_type, mode, device=device, infer=infer)])

        if self.infer:
            self.pos_encoder = skip_init(nn.Embedding, seq_len, self.embedding_dim, device=device)
        else:
            self.pos_encoder =  nn.Embedding(seq_len, self.embedding_dim, device=device)

        self.KANsformer_layers = nn.ModuleList([KANsformerLayer(self.embedding_dim, expand_dim, head_dim, num_heads, grid_min, grid_max, num_grids, base_activation, spline_weight_init_scale, use_layernorm, denominator, 0, radial_type, use_tanh, mode, gating, mask, device=device, infer=infer) for _ in range(num_layers)])

        out_in_dim = self.embedding_dim + added_output_dim

        output_dims = [out_in_dim, math.ceil(out_in_dim *0.3), math.ceil(out_in_dim *0.5), 32, 1]

        self.output_layers = nn.ModuleList([BuildLayer(in_dim, out_dim, grid_max, grid_min, num_grids_decode, True, base_activation, spline_weight_init_scale, denominator, 0, False, radial_type, mode, device=device, infer=infer) for in_dim, out_dim in zip(output_dims[:-1], output_dims[1:])])

        if residual:
            self.residuals = nn.ParameterList([nn.Parameter(torch.ones(self.embedding_dim, device=device)) for _ in range(num_layers)])

        if dropout_rate:
            self.Dropout = nn.Dropout(dropout_rate)
        else:
            self.Dropout = None

    def forward(self, x, infer=None):
        
        input_feature = np.ones(x.shape[-1], dtype=bool)

        if self.time_choice_feature is not None:
            barra_section = x[..., -self.barra_dim:]
            time_choice_feature = x[..., self.time_choice_feature]
            input_feature &= ~self.time_choice_feature
            time_choice_result = self.time_choice_block(time_choice_feature)

        if self.feature_insert is not None:
            insert_feature = x[..., self.feature_insert]
            input_feature &= ~self.feature_insert
        
        if self.trident:
            trident_tips = [x[..., trident_head]  for trident_head in self.trident_heads]        
        else:
            x = x[..., input_feature]
        
        batch_size, seq_len, _ = x.shape

        if self.Dropout is not None:
            if self.trident:
                trident_tips = [self.Dropout(x) for x in trident_tips]
            else:
                x = self.Dropout(x)

        if self.trident:
            for i in range(len(trident_tips)):
                # j = 0
                for layer in self.spear_heads[i]:
                    if self.use_tanh:
                        trident_tips[i] = self.tanh_norm * torch.tanh(trident_tips[i]/self.tanh_denom)
                    # print(i, j, trident_tips[i].shape)
                    # j += 1
                    trident_tips[i] = layer(trident_tips[i])
            if self.trident_merge_method == "concat":
                x = torch.cat(trident_tips, dim=-1)
                if self.use_tanh:
                    x = self.tanh_norm * torch.tanh(x/self.tanh_denom)
                x = self.embedding(x)
            elif self.trident_merge_method == "EW_ensemble":
                x = torch.zeros_like(trident_tips[0], dtype=trident_tips[0].dtype, device=trident_tips[0].device)
                for trident_tip in trident_tips:
                    x += trident_tip
        else:
            for layer in self.Embedding:
                if self.use_tanh:
                    x = self.tanh_norm * torch.tanh(x/self.tanh_denom)
                x = layer(x)

        pos_ebd = torch.arange(seq_len, device=x.device).unsqueeze(0).repeat(batch_size, 1)
        x = x + self.pos_encoder(pos_ebd)

        if self.residual:
            embed = x

        if infer is None:
            infer = self.infer

        for i in range(len(self.KANsformer_layers)):
            x = self.KANsformer_layers[i](x, infer)
            if self.residual:
                # print(x.shape, self.residuals[i].shape, embed.shape)
                x = x + self.residuals[i] * embed

        # print(insert_feature.shape, x.shape)

        if self.feature_insert is not None:
            x = torch.cat([x, insert_feature], dim=-1)

        for layer in self.output_layers:
            if self.use_tanh:
                x = self.tanh_norm * torch.tanh(x/self.tanh_denom)
            x = layer(x)

        if self.time_choice_feature is not None:            
            a = torch.linalg.vecdot(time_choice_result, barra_section, dim=2).unsqueeze(2)
            x += a

        ret = x.squeeze(-1)

        return ret


    # def forward_infer(self, x):

    #     batch_size, seq_len, _ = x.shape

    #     # if self.use_tanh:
    #     #     x = torch.tanh(x)

    #     x = self.Embedding(x)

    #     pos_ebd = torch.arange(seq_len, device=x.device).unsqueeze(0).repeat(batch_size, 1)
    #     x = x + self.pos_encoder(pos_ebd)

    #     for layer in self.KANsformer_layers:
    #         x = layer(x,True)

    #     ret = self.output_layer(x).squeeze(-1)
    #     return ret
