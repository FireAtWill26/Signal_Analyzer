from .fastkan import *
import torch

def BuildLayer(in_dim, out_dim, grid_min, grid_max, num_grids, use_base_update, base_activation, spline_weight_init_scale, denominator, dropout_rate, use_layernorm, radial_type, mode, device="cpu", infer=False):
    if mode == "KAF":
        layer = FastKAFLayer(in_dim, out_dim, num_grids, use_layernorm, dropout_rate, base_activation, spline_weight_init_scale, device=device, infer=infer)
    elif mode == "Cheby":
        layer = ChebyKANLayer(in_dim, out_dim, num_grids, spline_weight_init_scale, dropout_rate, device=device, infer=infer)
    else:
        layer = FastKANLayer(in_dim, out_dim, grid_min, grid_max, num_grids, use_base_update, use_layernorm, base_activation, spline_weight_init_scale, denominator, dropout_rate, radial_type, res_net=True, device=device, infer=infer)
    return layer

@MODEL_REGISTRY.register('KANGRU')
class KANGRU(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        grid_min = -2,
        grid_max = 2,
        num_grids = 16,
        num_layers = 1,
        use_base_update = True,
        base_activation = torch.nn.functional.gelu,
        spline_weight_init_scale = 1e-3,
        denominator = 0.33,
        use_layernorm = False,
        dropout_rate = 0.2,
        radial_type = "Gaussian",
        use_tanh = True,
        mode= "default",
        device = "cpu",
        infer = False,
    ):
        super(KANGRU, self).__init__()
        self.dropout_rate = 0
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.use_tanh = use_tanh
        self.device = device
        
        self.layers = nn.ModuleList()

        for idx in range(num_layers):

            if idx == 0:
                in_dim = input_dim
            else:
                in_dim = hidden_dim

            concat_dim = in_dim + hidden_dim

            LayerDict = nn.ModuleDict()
            if mode == "Cheby":
                LayerDict["kan_z"] = ChebyKANLayer(concat_dim, hidden_dim, num_grids, spline_weight_init_scale, self.dropout_rate, device=device, infer=infer)
                LayerDict["kan_r"] = ChebyKANLayer(concat_dim, hidden_dim, num_grids, spline_weight_init_scale, self.dropout_rate, device=device, infer=infer)
                LayerDict["cand_h"] = ChebyKANLayer(concat_dim, hidden_dim, num_grids, spline_weight_init_scale, self.dropout_rate, device=device, infer=infer)
            elif mode == "KAF":
                LayerDict["kan_z"] = FastKAFLayer(concat_dim, hidden_dim, num_grids, use_layernorm, self.dropout_rate, base_activation, spline_weight_init_scale, device=device, infer=infer)
                LayerDict["kan_r"] = FastKAFLayer(concat_dim, hidden_dim, num_grids, use_layernorm, self.dropout_rate, base_activation, spline_weight_init_scale, device=device, infer=infer)
                LayerDict["cand_h"] = FastKAFLayer(concat_dim, hidden_dim, num_grids, use_layernorm, self.dropout_rate, base_activation, spline_weight_init_scale, device=device, infer=infer)
            else:
                LayerDict["kan_z"] = FastKANLayer(concat_dim, hidden_dim, grid_min, grid_max, num_grids, use_base_update, use_layernorm, base_activation, spline_weight_init_scale, denominator, self.dropout_rate, radial_type, device=device, infer=infer)
                LayerDict["kan_r"] = FastKANLayer(concat_dim, hidden_dim, grid_min, grid_max, num_grids, use_base_update, use_layernorm, base_activation, spline_weight_init_scale, denominator, self.dropout_rate, radial_type, device=device, infer=infer)
                LayerDict["cand_h"] = FastKANLayer(concat_dim, hidden_dim, grid_min, grid_max, num_grids, use_base_update, use_layernorm, base_activation, spline_weight_init_scale, denominator, self.dropout_rate, radial_type, device=device, infer=infer)
            

            LayerDict["dropout"] = nn.Dropout(dropout_rate)

            self.layers.append(LayerDict)
        
        if mode == "Cheby":
            self.out = ChebyKANLayer(hidden_dim, 1, num_grids, spline_weight_init_scale, self.dropout_rate, device=device, infer=infer)
        elif mode == "KAF":
            self.out = FastKAFLayer(hidden_dim, 1, num_grids, use_layernorm, self.dropout_rate, base_activation, spline_weight_init_scale, device=device, infer=infer)
        else:
            self.out = FastKANLayer(hidden_dim, 1, grid_min, grid_max, num_grids, use_base_update, use_layernorm, base_activation, spline_weight_init_scale, denominator, self.dropout_rate, radial_type, device=device, infer=infer)

    def forward(self, input, h0=None, infer=None):

        batch_size, seq_len, input_dim = input.shape

        if h0 is None:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=input.device)

        current_input = input

        layer_hidden = []

        for idx in range(self.num_layers):
            layer = self.layers[idx]
            kan_z = layer["kan_z"]
            kan_r = layer["kan_r"]
            cand_h = layer["cand_h"]
            dropout = layer["dropout"]

            h_prev = h0[idx,:,:]

            current_hidden_states = []

            for t in range(seq_len):

                x_t = current_input[:,t,:]

                if self.use_tanh:
                    x_t = 2 * torch.tanh(x_t)
                    h_prev = 2 * torch.tanh(h_prev)

                concat = torch.cat([h_prev, x_t], dim = 1)
                # print(concat.shape)

                z_t = torch.sigmoid(kan_z(concat))
                r_t = torch.sigmoid(kan_r(concat))

                h_reset = r_t * h_prev
                concat_h = torch.cat([h_reset, x_t], dim=1)
                h_tilde = torch.tanh(cand_h(concat_h))

                h_t = (1 - z_t) * h_prev + z_t * h_tilde

                current_hidden_states.append(h_t)

                h_prev = h_t

            current_hidden_seq = torch.stack(current_hidden_states, dim=1)    
            
            if idx != self.num_layers - 1:
                current_hidden_seq = dropout(current_hidden_seq)

            current_input = current_hidden_seq
            layer_hidden.append(current_hidden_seq)

            output = self.out(current_input).squeeze(-1)

            return output
