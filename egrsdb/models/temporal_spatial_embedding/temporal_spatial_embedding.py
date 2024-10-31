import numpy as np
import torch
from absl.logging import error
from einops import rearrange, reduce, repeat
from torch import nn
from torch.nn import functional as F

from egrsdb.models.temporal_spatial_embedding.liif_spatial_embedding import LIIFDecoderAsSpatialEmbedding
from egrsdb.models.temporal_spatial_embedding.spatial_embedding import LargeViewFieldSpatialEmbedding


class SinCosTemporalEmbedding(nn.Module):
    def __init__(self, pe_dim):
        super(SinCosTemporalEmbedding, self).__init__()
        assert pe_dim % 2 == 0

        self.pe_sigma = 2048
        self.pe_dim = pe_dim

    def convert_posenc(self, x):
        # x: NCHW -> NHWC
        x = x.permute(0, 2, 3, 1)
        w = torch.exp(torch.linspace(0, np.log(self.pe_sigma), self.pe_dim // 2, device=x.device))
        x = torch.matmul(x.unsqueeze(-1), w.unsqueeze(0)).view(*x.shape[:-1], -1)
        x = torch.cat([torch.cos(np.pi * x), torch.sin(np.pi * x)], dim=-1)
        # x: NHWC -> NCHW
        x = x.permute(0, 3, 1, 2)
        return x

    def forward(self, t):
        t = self.convert_posenc(t)
        return t


class TemporalEmbedding(nn.Module):
    def __init__(self, inr_in_channel, inr_our_channel, embedding_type, time_dim_increasing_type):
        super(TemporalEmbedding, self).__init__()
        assert embedding_type in ["add", "mul"]
        assert time_dim_increasing_type in ["two_mlp_learning", "sin_cos_encoding"]

        self.inr_channel = inr_in_channel
        self.embedding_type = embedding_type
        self.time_dim_increasing_type = time_dim_increasing_type

        if time_dim_increasing_type == "two_mlp_learning":
            self.temporal_embedding_1 = nn.Conv2d(1, inr_in_channel, 1, padding=0)
            self.temporal_embedding_2 = nn.Conv2d(inr_in_channel + 1, inr_in_channel, 1, padding=0)
        else:
            self.temporal_embedding = SinCosTemporalEmbedding(pe_dim=inr_in_channel)
        self.condense_temporal_embedding = nn.Conv2d(inr_in_channel, inr_our_channel, 1, padding=0)

    def forward(self, gloabl_inr, embedded_time):
        B, C, H, W = gloabl_inr.shape
        time_map = torch.zeros(1, H, W, B).to(gloabl_inr.device) + embedded_time
        time_map = rearrange(time_map, "c h w b -> b c h w")

        if self.time_dim_increasing_type == "two_mlp_learning":
            temporal_embedding_1 = self.temporal_embedding_1(time_map)
            temporal_cat_time = torch.cat([time_map, temporal_embedding_1], dim=1)
            temporal_embedding_2 = self.temporal_embedding_2(temporal_cat_time)
        else:
            temporal_embedding_2 = self.temporal_embedding(time_map)

        if self.embedding_type == "add":
            gloabl_inr = gloabl_inr + temporal_embedding_2
        else:
            gloabl_inr = gloabl_inr * temporal_embedding_2
        gloabl_inr = self.condense_temporal_embedding(gloabl_inr)
        # B, C, H, W -> B, C, 1, 1
        temporal_embedding_2_one_dim = torch.mean(temporal_embedding_2, dim=[2, 3], keepdim=True)
        return gloabl_inr, temporal_embedding_2_one_dim


class BilinearSpatialEmbedding(nn.Module):
    def __init__(self, inr_temporal_out_channel):
        super(BilinearSpatialEmbedding, self).__init__()
        self.inr_temporal_out_channel = inr_temporal_out_channel
        hidden_channels = 128
        out_channels = 3
        self.decoding = nn.Sequential(
            nn.Conv2d(inr_temporal_out_channel, hidden_channels, kernel_size=1, stride=1, padding=0),
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

    def forward(self, inr, sH, sW):
        sr_h, sr_w = sH, sW
        inr = F.interpolate(inr, size=(sr_h, sr_w), mode="bicubic", align_corners=False)
        output = self.decoding(inr)
        return output


class NoneSpatialEmbeddingOnlyDecoding(nn.Module):
    def __init__(self, inr_temporal_out_channel):
        super(NoneSpatialEmbeddingOnlyDecoding, self).__init__()
        self.inr_temporal_out_channel = inr_temporal_out_channel
        hidden_channels = 128
        out_channels = 3
        self.decoding = nn.Sequential(
            nn.Conv2d(inr_temporal_out_channel, hidden_channels, kernel_size=1, stride=1, padding=0),
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

    def forward(self, inr, sH, sW):
        sr_h, sr_w = sH, sW
        B, C, H, W = inr.shape
        if H != sr_h or W != sr_w:
            error(
                f"NoneSpatialEmbeddingOnlyDecoding not support Up Sampling. sr_h: {sr_h}, sr_w: {sr_w}, H: {H}, W: {W}"
            )
            exit(1)
        output = self.decoding(inr)
        return output


class TemporalSpatialEmbedding(nn.Module):
    def __init__(
        self,
        inr_temporal_in_channel,
        inr_temporal_out_channel,
        temporal_embedding_type,
        spatial_embedding_type,
        spatial_embedding_config,
    ):
        super(TemporalSpatialEmbedding, self).__init__()

        self.temporal_embedding = TemporalEmbedding(
            inr_temporal_in_channel,
            inr_temporal_out_channel,
            temporal_embedding_type,
            spatial_embedding_config.time_dim_increasing_type,
        )

        if spatial_embedding_type == "large_view_field_spatial_decoder":
            self.spatial_embedding = LargeViewFieldSpatialEmbedding(
                spatial_embedding_config.spatial_coords_dim,
                inr_temporal_out_channel,
                spatial_embedding_config.query_kernel_size,
                spatial_embedding_config.query_kernel_type,
                spatial_embedding_config.query_kernel_dilation_rate,
            )
        elif spatial_embedding_type == "bicubic_spatial_decoder":
            self.spatial_embedding = BilinearSpatialEmbedding(inr_temporal_out_channel)
        elif spatial_embedding_type == "LIIF_spatial_decoder":
            self.spatial_embedding = LIIFDecoderAsSpatialEmbedding(inr_temporal_out_channel)
        elif spatial_embedding_type == "none_spatial_decoder":
            self.spatial_embedding = NoneSpatialEmbeddingOnlyDecoding(inr_temporal_out_channel)
        else:
            raise NotImplementedError(f"spatial_embedding_type: {spatial_embedding_type}")

    def forward(self, inr, t, sH, sW):
        t_inr, time_embed_feature = self.temporal_embedding(inr, t)
        t_sh_sw_inr = self.spatial_embedding(t_inr, sH, sW)
        return t_sh_sw_inr, time_embed_feature
