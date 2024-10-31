import math

import einops
import numpy as np
import torch
import torch.nn.functional as F
from absl.logging import info
from torch import nn

from egrsdb.models.transformer.transformer import TransformerEncoder


def init_wb(shape):
    weight = torch.empty(shape[1], shape[0] - 1)
    nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
    bias = torch.empty(shape[1], 1)
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
    nn.init.uniform_(bias, -bound, bound)
    return torch.cat([weight, bias], dim=1).t().detach()


def make_pair(x):
    if isinstance(x, int):
        return (x, x)
    return x


def batched_linear_mm(x, wb):
    # x: (B, N, D1); wb: (B, D1 + 1, D2) or (D1 + 1, D2)
    one = torch.ones(*x.shape[:-1], 1, device=x.device)
    return torch.matmul(torch.cat([x, one], dim=-1), wb)


class Tokenizer(nn.Module):
    def __init__(self, input_size, patch_size, dim, padding, in_channels):
        super().__init__()
        input_size = make_pair(input_size)
        patch_size = make_pair(patch_size)
        padding = make_pair(padding)
        self.patch_size = patch_size
        self.padding = padding
        self.prefc = nn.Linear(patch_size[0] * patch_size[1] * in_channels, dim)
        n_patches = ((input_size[0] + padding[0] * 2) // patch_size[0]) * (
            (input_size[1] + padding[1] * 2) // patch_size[1]
        )
        self.posemb = nn.Parameter(torch.randn(n_patches, dim))

    def forward(self, x):
        p = self.patch_size
        x = F.unfold(x, p, stride=p, padding=self.padding)  # (B, C * p * p, L)
        x = x.permute(0, 2, 1).contiguous()
        x = self.prefc(x) + self.posemb.unsqueeze(0)
        return x


class InrMlp(nn.Module):
    def __init__(self, depth, in_dim, out_dim, hidden_dim, use_pe, pe_dim, out_bias=0, pe_sigma=1024):
        super().__init__()
        self.use_pe = use_pe
        self.pe_dim = pe_dim
        self.pe_sigma = pe_sigma
        self.depth = depth
        self.param_shapes = dict()
        if use_pe:
            last_dim = in_dim * pe_dim
        else:
            last_dim = in_dim
        for i in range(depth):
            cur_dim = hidden_dim if i < depth - 1 else out_dim
            self.param_shapes[f"wb{i}"] = (last_dim + 1, cur_dim)
            info(f"wb{i}: {last_dim + 1} -> {cur_dim}")
            last_dim = cur_dim
        self.relu = nn.ReLU()
        self.params = None
        self.out_bias = out_bias

    def set_params(self, params):
        self.params = params

    def convert_posenc(self, x):
        w = torch.exp(torch.linspace(0, np.log(self.pe_sigma), self.pe_dim // 2, device=x.device))
        x = torch.matmul(x.unsqueeze(-1), w.unsqueeze(0)).view(*x.shape[:-1], -1)
        x = torch.cat([torch.cos(np.pi * x), torch.sin(np.pi * x)], dim=-1)
        return x

    def forward(self, x):
        # B, C, H, W -> B, H, W, C
        x = x.permute(0, 2, 3, 1).contiguous()
        B, query_shape = x.shape[0], x.shape[1:-1]
        x = x.reshape(B, -1, x.shape[-1])
        if self.use_pe:
            x = self.convert_posenc(x)
        for i in range(self.depth):
            x = batched_linear_mm(x, self.params[f"wb{i}"])
            if i < self.depth - 1:
                x = self.relu(x)
            else:
                x = x + self.out_bias
        x = x.view(B, *query_shape, -1)
        # B, H, W, C -> B, C, H, W
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class TransformerMLPParams(nn.Module):
    def __init__(
        self,
        image_width,
        image_height,
        patch_size,
        image_channel,
        event_in_channel,
        token_padding,
        transformer_dim,
        transformer_depth,
        position_embedding,
        n_head,
        head_dim,
        ff_dim,
        dropout,
        inr_depth,
        inr_in_dim,
        inr_out_dim,
        inr_hidden_dim,
        n_groups,
    ):
        super().__init__()
        input_size = (image_width, image_height)
        in_channels = image_channel + event_in_channel
        self.tokenizer = Tokenizer(input_size, patch_size, transformer_dim, token_padding, in_channels)
        self.transformer = TransformerEncoder(transformer_dim, transformer_depth, n_head, head_dim, ff_dim, dropout)
        position_embedding_dim = 128
        self.mlp_inr = InrMlp(
            depth=inr_depth,
            in_dim=inr_in_dim,
            out_dim=inr_out_dim,
            hidden_dim=inr_hidden_dim,
            use_pe=position_embedding,
            pe_dim=position_embedding_dim,
        )
        # params
        self.base_params = nn.ParameterDict()
        n_wtokens = 0
        self.wtoken_postfc = nn.ModuleDict()
        self.wtoken_rng = dict()
        for name, shape in self.mlp_inr.param_shapes.items():
            self.base_params[name] = nn.Parameter(init_wb(shape))
            g = min(n_groups, shape[1])
            assert shape[1] % g == 0
            self.wtoken_postfc[name] = nn.Sequential(
                nn.LayerNorm(transformer_dim),
                nn.Linear(transformer_dim, shape[0] - 1),
            )
            self.wtoken_rng[name] = (n_wtokens, n_wtokens + g)
            n_wtokens += g
        self.wtokens = nn.Parameter(torch.randn(n_wtokens, transformer_dim))

    def forward(self, events, blur_frame):
        event_rsb = torch.cat([events, blur_frame], dim=1)
        B, C, H, W = event_rsb.shape
        dtokens = self.tokenizer(event_rsb)
        wtokens = einops.repeat(self.wtokens, "n d -> b n d", b=B)
        trans_out = self.transformer(torch.cat([dtokens, wtokens], dim=1))
        trans_out = trans_out[:, -len(self.wtokens) :, :]

        params = dict()
        for name, shape in self.mlp_inr.param_shapes.items():
            wb = einops.repeat(self.base_params[name], "n m -> b n m", b=B)
            w, b = wb[:, :-1, :], wb[:, -1:, :]
            l, r = self.wtoken_rng[name]
            x = self.wtoken_postfc[name](trans_out[:, l:r, :])
            x = x.transpose(-1, -2)  # (B, shape[0] - 1, g)
            w = F.normalize(w * x.repeat(1, 1, w.shape[2] // x.shape[2]), dim=1)

            wb = torch.cat([w, b], dim=1)
            params[name] = wb

        self.mlp_inr.set_params(params)
        return self.mlp_inr
