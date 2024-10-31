import torch


def up_scale(s, H, W):
    return int(s * H), int(s * W)


def make_coord(shape, flatten=True):
    """Make coordinates at grid centers."""
    coord_seqs = []
    for i, n in enumerate(shape):
        if isinstance(n, torch.Tensor):
            n = int(n.item())
        v0, v1 = -1, 1
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret
