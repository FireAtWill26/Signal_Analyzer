from .fastkan import *
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
        ffn_output = self.feed_forward(x)
        if self.use_layernorm:
            x = self.norm2(x + self.dropout2(ffn_output))
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
        mode = "default",
        mask: bool = False,
        seq_len = 20,
        gating: bool = True,
        residual: bool = False,
        feature_insert = None,
        device = "cpu",
        infer = False, ):
        super().__init__()

        # self.use_tanh = use_tanh
        self.infer = infer
        self.residual = residual
        self.use_tanh = use_tanh

        self.feature_insert = feature_insert

        added_dim = 0 if feature_insert is None else feature_insert.sum()

        input_dim -= added_dim

        if type(embedding_dim) == int:
            self.embedding_dim = embedding_dim
            self.Embedding = nn.ModuleList([BuildLayer(input_dim, embedding_dim, grid_max, grid_min, num_grids, True, base_activation, spline_weight_init_scale, denominator, 0, use_layernorm, radial_type, mode, device=device, infer=infer)])
        if isinstance(embedding_dim, list):
            self.embedding_dim = embedding_dim[-1]
            if len(embedding_dim) > 1:
                embedding_layers = []
                for i in range(len(embedding_dim) - 1):
                    embedding_layers.append(math.ceil(embedding_dim[i] * input_dim))
                embedding_layers.append(self.embedding_dim)
                # print(embedding_layers)
                self.Embedding = nn.ModuleList([BuildLayer(in_dim, out_dim, grid_max, grid_min, num_grids_embed, True, base_activation, spline_weight_init_scale, denominator, 0, use_layernorm, radial_type, mode, device=device, infer=infer) for in_dim, out_dim in zip(embedding_layers[:-1], embedding_layers[1:])])
            else:
                self.Embedding = nn.ModuleList([BuildLayer(input_dim, self.embedding_dim, grid_max, grid_min, num_grids_embed, True, base_activation, spline_weight_init_scale, denominator, 0, use_layernorm, radial_type, mode, device=device, infer=infer)])

        if self.infer:
            self.pos_encoder = skip_init(nn.Embedding, seq_len, self.embedding_dim, device=device)
        else:
            self.pos_encoder =  nn.Embedding(seq_len, self.embedding_dim, device=device)

        self.KANsformer_layers = nn.ModuleList([KANsformerLayer(self.embedding_dim, expand_dim, head_dim, num_heads, grid_min, grid_max, num_grids, base_activation, spline_weight_init_scale, use_layernorm, denominator, 0, radial_type, use_tanh, mode, gating, mask, device=device, infer=infer) for _ in range(num_layers)])

        out_in_dim = self.embedding_dim + added_dim

        output_dims = [out_in_dim, math.ceil(out_in_dim *0.3), math.ceil(out_in_dim *0.5), 32, 1]

        self.output_layers = nn.ModuleList([BuildLayer(in_dim, out_dim, grid_max, grid_min, num_grids_decode, True, base_activation, spline_weight_init_scale, denominator, 0, use_layernorm, radial_type, mode, device=device, infer=infer) for in_dim, out_dim in zip(output_dims[:-1], output_dims[1:])])

        if residual:
            self.residuals = [nn.Parameter(torch.randn(self.embedding_dim, device=device)) for _ in range(num_layers)]

        if dropout_rate:
            self.Dropout = nn.Dropout(dropout_rate)
        else:
            self.Dropout = None

    def forward(self, x, infer=None):

        if self.feature_insert is not None:
            insert_feature = x[..., self.feature_insert]
            x = x[..., ~self.feature_insert]
        
        batch_size, seq_len, _ = x.shape


        if self.Dropout is not None:
            x = self.Dropout(x)

        for layer in self.Embedding:
            if self.use_tanh:
                x = torch.tanh(x)
            x = layer(x)

        pos_ebd = torch.arange(seq_len, device=x.device).unsqueeze(0).repeat(batch_size, 1)
        x = x + self.pos_encoder(pos_ebd)

        if self.residual:
            embed = x

        if infer is None:
            infer = self.infer

        for i in range(len(self.KANsformer_layers)):
            if self.use_tanh:
                x = torch.tanh(x)
            x = self.KANsformer_layers[i](x, infer)
            if self.residual:
                # print(x.shape, self.residuals[i].shape, embed.shape)
                x = x + self.residuals[i] * embed

        # print(insert_feature.shape, x.shape)

        if self.feature_insert is not None:
            x = torch.cat([x, insert_feature], dim=-1)

        for layer in self.output_layers:
            if self.use_tanh:
                x = torch.tanh(x)
            x = layer(x)

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