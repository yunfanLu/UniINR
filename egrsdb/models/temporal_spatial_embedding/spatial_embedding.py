import torch
import torch.nn.functional as F
from absl.logging import info
from einops import rearrange, reduce, repeat
from torch import nn

from egrsdb.utils.coords import make_coord, up_scale

DEBUG = True

# Using large view field for spatial embedding.


class LargeViewFieldSpatialEmbedding(nn.Module):
    def __init__(
        self, coords_dim, global_inr_channel, query_kernel_size, query_kernel_type, query_kernel_dilation_rate=2
    ):
        """
        coords_dim: 2, the position dim.
        global_inr_channel: the channel of global inr.
        query_kernel_size: each query kernel size.
        """
        super(LargeViewFieldSpatialEmbedding, self).__init__()
        assert query_kernel_size >= 3 and query_kernel_size % 2 == 1, f"query_kernel_size: {query_kernel_size}"
        assert query_kernel_dilation_rate >= 2, f"query_kernel_dilation_rate: {query_kernel_dilation_rate}"
        # config
        self.coords_dim = coords_dim
        self.global_inr_channel = global_inr_channel
        self.query_kernel_size = query_kernel_size
        self.query_kernel_type = query_kernel_type
        self.query_kernel_dilation_rate = query_kernel_dilation_rate
        # model
        query_inr_coord_with_range_dim = coords_dim * query_kernel_size * query_kernel_size + 2
        embedding_inr_channel = global_inr_channel * query_kernel_size * query_kernel_size
        self.spatial_coord_embedding = nn.Sequential(
            nn.Conv2d(query_inr_coord_with_range_dim, embedding_inr_channel, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            nn.Conv2d(embedding_inr_channel, embedding_inr_channel, kernel_size=1, stride=1, padding=0),
        )

        self.rgb_decoder = nn.Sequential(
            nn.Conv2d(embedding_inr_channel, 128, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            nn.Conv2d(128, 3, kernel_size=1, stride=1, padding=0),
        )
        self._info()

    def forward(self, inr_feature, sH, sW):
        """
        Sampling in the inr_feature, and embedding the spatial information to decode the rgb image with resolution
        (sH, sW).
        inr_feature: [B, C, H, W]
        sH, sW: the output resolution.
        """
        B, C, H, W = inr_feature.shape
        # the coord distance in the inr_feature.
        rx, ry = 1.0 / H, 1.0 / W
        # (sH, sW) -> [sH, sW, 2]
        sr_coord_absolute = make_coord((sH, sW), flatten=False).to(inr_feature.device)
        # sr coord as the query coord. [sH, sW, qks * qks, 2]
        query_sr_coord = self._sr_coord_to_query_coord(sr_coord_absolute)
        # [sH, sW, qks * qks, 2] -> [B, sH, sW, qks * qks, 2]
        query_sr_coord = repeat(query_sr_coord, "sH sW qks_qks c -> b sH sW qks_qks c", b=B)
        query_sr_coord = rearrange(query_sr_coord, "b sH sW q c -> b (sH sW) q c")

        # 2. The query coord is ready, now we need to sample in the inr_feature and inr_feature coord.
        # 2.1 inr_feature_coord first.
        inr_feature_coord = make_coord((H, W), flatten=False).to(inr_feature.device)  # [H, W, 2]
        inr_feature_coord = repeat(inr_feature_coord, "h w c -> b h w c", b=B)  # [B, H, W, 2]
        inr_feature_coord = rearrange(inr_feature_coord, "b h w c -> b c h w")  # [B, 2, H, W]
        # 2.1 (1) shape of 'query_inr_coord' and 'query_inr_feature': [B, 2, sH * sW, qks * qks]
        query_inr_coord = F.grid_sample(inr_feature_coord, query_sr_coord.flip(-1), mode="nearest", align_corners=False)
        query_inr_coord = rearrange(query_inr_coord, "b c sH_sW qks_qks -> b sH_sW qks_qks c")
        query_inr_coord_relatively = query_inr_coord - query_sr_coord
        # [B, sH * sW, qks * qks, C] -> [B, sH * sW, qks * qks * C]
        query_inr_coord_relatively = rearrange(query_inr_coord_relatively, "b sH_sW qks_qks c -> b sH_sW (qks_qks c)")
        # [B, sH * sW, qks * qks * C] -> [B, sH, sW, qks * qks * C]
        query_inr_coord_relatively = rearrange(
            query_inr_coord_relatively, "b (sH sW) qks_qks_c -> b sH sW qks_qks_c", sH=sH, sW=sW
        )

        # 2.2 inr_feature second.
        query_inr_feature = F.grid_sample(inr_feature, query_sr_coord.flip(-1), mode="nearest", align_corners=False)
        query_inr_feature = rearrange(query_inr_feature, "b c sH_sW qks_qks -> b sH_sW qks_qks c")
        query_inr_feature = rearrange(query_inr_feature, "b (sH sW) qks_qks c -> b sH sW (qks_qks c)", sH=sH, sW=sW)
        query_inr_feature = rearrange(query_inr_feature, "b h w c -> b c h w")

        # 3. mapping query coord to embedding space
        query_range_size = torch.zeros(B, sH, sW, 2) + torch.tensor([rx, ry])
        query_range_size = query_range_size.to(inr_feature.device)
        # [B, sH, sW, qks * qks * 2 + 2]
        query_inr_coord_relatively_with_range = torch.cat([query_inr_coord_relatively, query_range_size], dim=-1)
        query_inr_coord_relatively_with_range = rearrange(
            query_inr_coord_relatively_with_range, "b sH sW qks_qks_c -> b qks_qks_c sH sW"
        )
        query_inr_coord_embedding = self.spatial_coord_embedding(query_inr_coord_relatively_with_range)

        # 4. decode the rgb image.
        embedding_feature = query_inr_feature * query_inr_coord_embedding
        rgb_frame = self.rgb_decoder(embedding_feature)
        return rgb_frame

    def _sr_coord_to_query_coord(self, sr_coord_absolute):
        """generate query coord from sr coord.
        sr_coord_absolute: [B, sH, sW, 2]
        return: [B, sH, sW, query_kernel_size * query_kernel_size, 2]
        """
        qks = self.query_kernel_size
        query_coord_absolure = sr_coord_absolute.unsqueeze(-2).repeat(1, 1, qks * qks, 1)
        dxs, dys = self._get_dynamic_kernel_bais()
        dxs = dxs.reshape(qks * qks).to(sr_coord_absolute.device)
        dys = dys.reshape(qks * qks).to(sr_coord_absolute.device)

        query_coord_absolure[:, :, :, 0] = query_coord_absolure[:, :, :, 0] + dxs
        query_coord_absolure[:, :, :, 1] = query_coord_absolure[:, :, :, 0] + dxs
        return query_coord_absolure

    def _get_dynamic_kernel_bais(self):
        if self.query_kernel_type == "uniform":
            # N x N conv
            dxy = torch.arange(self.query_kernel_size) - self.query_kernel_size // 2
            dxs, dys = dxy, dxy
        elif self.query_kernel_type == "dilated":
            # N x N dilated conv
            dxy = torch.arange(self.query_kernel_size) - self.query_kernel_size // 2
            dxs = dxs * self.query_kernel_dilation_rate
            dys = dys * self.query_kernel_dilation_rate
        # 这个地方可以再扩展，比如曼哈顿距离等。
        else:
            raise NotImplementedError
        dxs, dys = torch.meshgrid(dxs, dys)
        return dxs, dys

    def _info(self):
        info(f"LargeKernelSpatialEmbedding:")
        info(f"  coords_dim: {self.coords_dim}")
        info(f"  global_inr_channel: {self.global_inr_channel}")
        info(f"  query_kernel_size: {self.query_kernel_size}")
        info(f"  query_kernel_type: {self.query_kernel_type}")
        info(f"  query_kernel_dilation_rate: {self.query_kernel_dilation_rate}")
