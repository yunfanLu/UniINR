import torch
import torch.nn.functional as F
from absl.logging import info
from einops import rearrange, reduce, repeat
from torch import nn

from egrsdb.utils.coords import make_coord, up_scale


class VFISRConv1x1DecoderLearnedPositionEmbedding(nn.Module):
    def __init__(self, coords_dim, global_inr_channel, local_inr_channel, hidden_channels, out_channels) -> None:
        super().__init__()
        self.position_encoder = nn.Sequential(
            nn.Conv2d(coords_dim, global_inr_channel, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            nn.Conv2d(global_inr_channel, global_inr_channel, kernel_size=1, stride=1, padding=0),
        )

        self.global_inr_con1x1 = nn.Sequential(
            nn.Conv2d(global_inr_channel, hidden_channels, kernel_size=1, stride=1, padding=0), nn.GELU()
        )
        self.event_tree_con1x1 = nn.Sequential(
            nn.Conv2d(local_inr_channel, hidden_channels, kernel_size=1, stride=1, padding=0), nn.GELU()
        )
        self.decoding = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
        )

    def forward(self, global_inr, query_coords, out_resolution):
        sr_h, sr_w = out_resolution
        global_inr = F.interpolate(global_inr, size=(sr_h, sr_w), mode="bilinear", align_corners=False)
        query_coords = self.position_encoder(query_coords)
        global_inr = global_inr + query_coords
        global_inr_focused = self.global_inr_con1x1(global_inr)
        # up sampling
        output = self.decoding(global_inr_focused)
        return output
