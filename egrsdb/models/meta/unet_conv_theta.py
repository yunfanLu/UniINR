import numpy as np
import torch
from absl.logging import info
from torch import nn
from torch.nn import functional as F

from egrsdb.models.els_net.esl_backbone import ESLBackBone
from egrsdb.models.unet.unet_2d import UNet
from egrsdb.models.unet.unet_2d_dcnv3 import UNetDVNv3


class Conv1x1DecoderLearnedPositionEmbedding(nn.Module):
    def __init__(self, coords_dim, in_channels, hidden_channels, out_channels, depth, time_embedding_type):
        super().__init__()
        assert time_embedding_type in ["ADD", "CONCAT", "MULTIPLY"]
        self.time_embedding_type = time_embedding_type
        # mapping coords to a high dimensional space with the same dimension as in_channels
        self.position_encoder = nn.Conv2d(coords_dim, in_channels, kernel_size=1, stride=1, padding=0)
        #
        self.model = nn.Sequential()
        if self.time_embedding_type == "CONCAT":
            in_channels += in_channels
        self.model.append(nn.Conv2d(in_channels, hidden_channels, kernel_size=1, stride=1, padding=0))
        self.model.append(nn.ReLU())
        for i in range(depth - 2):
            self.model.append(nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, stride=1, padding=0))
            self.model.append(nn.ReLU())
        self.model.append(nn.Conv2d(hidden_channels, out_channels, kernel_size=1, stride=1, padding=0))
        info(f"Conv1x1DecoderLearnedPositionEmbedding:")
        info(f"  time_embedding_type: {time_embedding_type}")
        info(f"  in_channels: {in_channels}")
        info(f"  hidden_channels: {hidden_channels}")
        info(f"  out_channels: {out_channels}")

    def set_theta(self, theta):
        self.theta = theta

    def forward(self, coords):
        feature_gs_t = self.position_encoder(coords)
        if self.time_embedding_type == "ADD":
            feature_gs_t = feature_gs_t + self.theta
        elif self.time_embedding_type == "MULTIPLY":
            feature_gs_t = feature_gs_t * self.theta
        elif self.time_embedding_type == "CONCAT":
            feature_gs_t = torch.cat([feature_gs_t, self.theta], dim=1)
        else:
            raise NotImplementedError
        frame_reconstructed = self.model(feature_gs_t)
        return frame_reconstructed



class UNetCon1x1WithTheta(nn.Module):
    def __init__(
        self,
        encoder_type,
        decoder_type,
        coords_dim,
        depth,
        meta_in_channels,
        meta_out_channels,
        inr_in_channel,
        inr_mid_channel,
        inr_out_channel,
        dcn_config,
        esl_config,
        time_embedding_type,
    ):
        super().__init__()
        assert encoder_type in ["unet", "unet_dcnv3", "esl_backbone"]
        assert decoder_type in [
            "conv1x1",
            "conv1x1_learn_position_encoding",
            "conv1x1_sin_cos_encoding",
            "conv1x1_multi_coords",
        ]

        self.meta_in_channels = meta_in_channels
        self.meta_out_channels = meta_out_channels
        self.inr_in_channel = inr_in_channel
        self.inr_mid_channel = inr_mid_channel
        self.inr_out_channel = inr_out_channel
        self.coords_dim = coords_dim
        self.depth = depth
        self.dcn_config = dcn_config
        self.esl_config = esl_config
        self.time_embedding_type = time_embedding_type

        self.encoder = self._get_encoder(encoder_type)
        self.e_inr = self._get_decoder(decoder_type)

    def _get_encoder(self, encoder_type):
        if encoder_type == "esl_backbone":
            return ESLBackBone(
                input_frames=1,
                is_color=self.esl_config.is_color,
                event_moments=self.esl_config.event_moments,
                hidden_channels=self.esl_config.hidden_channels,
                high_dim_channels=self.esl_config.high_dim_channels,
                is_deformable=self.esl_config.is_deformable,
                loop=self.esl_config.loop,
                has_scn_loop=self.esl_config.has_scn_loop,
            )
        else:
            raise NotImplementedError
        return model

    def _get_decoder(self, decoder_type):
        if decoder_type == "conv1x1_learn_position_encoding":
            model = Conv1x1DecoderLearnedPositionEmbedding(
                self.coords_dim,
                self.inr_in_channel,
                self.inr_mid_channel,
                self.inr_out_channel,
                self.depth,
                self.time_embedding_type,
            )
        return model

    def forward(self, events, blur_frame):
        inr_theta = self.encoder(events, blur_frame)
        self.e_inr.set_theta(inr_theta)
        return self.e_inr
