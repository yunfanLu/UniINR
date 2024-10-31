import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat

# LIIF decoder: liif/models/liif.py


def make_coord(shape, ranges=None, flatten=True):
    """Make coordinates at grid centers."""
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_list):
        super().__init__()
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ReLU())
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        shape = x.shape[:-1]
        x = self.layers(x.view(-1, x.shape[-1]))
        return x.view(*shape, -1)


class LIIFDecoderAsSpatialEmbedding(nn.Module):
    def __init__(self, inr_temporal_out_channel, local_ensemble=True, feat_unfold=True, cell_decode=True):
        super().__init__()
        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold
        self.cell_decode = cell_decode

        # self.encoder = models.make(encoder_spec)
        self.out_dim = 3
        self.hidden_list = [256, 256, 256, 256]

        self.imnet_in_dim = inr_temporal_out_channel

        if self.feat_unfold:
            self.imnet_in_dim *= 9
        self.imnet_in_dim += 2  # attach coord
        if self.cell_decode:
            self.imnet_in_dim += 2
        self.imnet = MLP(self.imnet_in_dim, self.out_dim, self.hidden_list)

    def query_rgb(self, feat, coord, cell=None):
        # coord: [bs, query_batch_size, 2], [1, 30000, 2]
        # cell: [bs, query_batch_size, 2], [1, 30000, 2], cell[i,j] = [1/h, 1/w]
        # feat = self.feat  # 1, 64, 625, 970. [bs, c, h, w]

        if self.imnet is None:
            ret = F.grid_sample(feat, coord.flip(-1).unsqueeze(1), mode="nearest", align_corners=False)[
                :, :, 0, :
            ].permute(0, 2, 1)
            return ret

        if self.feat_unfold:
            # after unfold: [bs, c*9, h, w], 1, 576, 625, 970
            feat = F.unfold(feat, 3, padding=1).view(feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])

        if self.local_ensemble:
            vx_lst = [-1, 1]
            vy_lst = [-1, 1]
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, eps_shift = [0], [0], 0

        # field radius (global: [-1, 1])
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        # feat_coord: [1, 2, h, w], [1, 2, 625, 970]
        feat_coord = (
            make_coord(feat.shape[-2:], flatten=False)
            .cuda()
            .permute(2, 0, 1)
            .unsqueeze(0)
            .expand(feat.shape[0], 2, *feat.shape[-2:])
        )

        preds = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                q_feat = F.grid_sample(feat, coord_.flip(-1).unsqueeze(1), mode="nearest", align_corners=False)
                q_feat = q_feat[:, :, 0, :].permute(0, 2, 1)  # 1, 30000, 576

                # 1, 30000, 2
                q_coord = F.grid_sample(feat_coord, coord_.flip(-1).unsqueeze(1), mode="nearest", align_corners=False)
                q_coord = q_coord[:, :, 0, :].permute(0, 2, 1)

                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]
                inp = torch.cat([q_feat, rel_coord], dim=-1)  # 1, 30000, 578. = 576 + 2

                if self.cell_decode:
                    rel_cell = cell.clone()
                    rel_cell[:, :, 0] *= feat.shape[-2]
                    rel_cell[:, :, 1] *= feat.shape[-1]
                    inp = torch.cat([inp, rel_cell], dim=-1)

                bs, q = coord.shape[:2]
                pred = self.imnet(inp.view(bs * q, -1)).view(bs, q, -1)
                preds.append(pred)

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)
        if self.local_ensemble:
            t = areas[0]
            areas[0] = areas[3]
            areas[3] = t
            t = areas[1]
            areas[1] = areas[2]
            areas[2] = t
        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)
        return ret

    def forward(self, inr, sH, sW):
        sH = int(sH)
        sW = int(sW)
        # coord: sH, sW, 2
        coord = make_coord((sH, sW), flatten=False)
        cell = torch.ones_like(coord)
        # coord and cell to inr drive.
        coord = coord.to(inr.device)
        cell = cell.to(inr.device)
        # Here it is 2 because make coord has -1 to 1
        cell[:, 0] *= 2 / sH
        cell[:, 1] *= 2 / sW
        B, C, H, W = inr.shape
        coord = repeat(coord, "h w c -> b h w c", b=B)
        cell = repeat(cell, "h w c -> b h w c", b=B)
        coord = rearrange(coord, "b h w c -> b (h w) c")
        cell = rearrange(cell, "b h w c -> b (h w) c")
        sr = self.query_rgb(inr, coord, cell)
        sr = rearrange(sr, "b (h w) c -> b h w c", h=sH, w=sW)
        sr = rearrange(sr, "b h w c -> b c h w")
        return sr
