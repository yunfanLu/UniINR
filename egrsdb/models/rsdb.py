import torch
from absl.logging import info
from torch import nn

from egrsdb.functions.rolling_coords import get_t_global_shutter_coordinate, get_t_rolling_shutter_coordinate
from egrsdb.functions.timestamp import get_gs_sharp_timestamps, get_rs_shape_timestamps
from egrsdb.models.meta.unet_conv_theta import UNetCon1x1WithTheta

DEBUG = False


def get_rolling_shutter_deblur_model(
    image_channel,
    coords_dim,
    events_moment,
    meta_type,
    encoder_name,
    decoder_name,
    inr_depth,
    inr_in_channel,
    inr_mid_channel,
    image_height,
    image_width,
    rs_blur_timestamp,
    gs_sharp_count,
    rs_integral,
    dcn_config,
    esl_config,
    correct_offset,
    time_embedding_type,
    intermediate_visualization=True,
):
    _, gs_sharp_timestamps = get_gs_sharp_timestamps(0, 520, gs_sharp_count, 520, correct_offset)
    rs_sharp_timestamps = get_rs_shape_timestamps(rs_blur_timestamp, rs_integral)
    model = RollingShutterDeblur(
        image_channel=image_channel,
        coords_dim=coords_dim,
        events_moment=events_moment,
        meta_type=meta_type,
        encoder_name=encoder_name,
        decoder_name=decoder_name,
        inr_depth=inr_depth,
        inr_in_channel=inr_in_channel,
        inr_mid_channel=inr_mid_channel,
        image_height=image_height,
        image_width=image_width,
        rs_blur_timestamp=rs_blur_timestamp,
        gs_sharp_timestamps=gs_sharp_timestamps,
        rs_sharp_timestamps=rs_sharp_timestamps,
        intermediate_visualization=intermediate_visualization,
        dcn_config=dcn_config,
        esl_config=esl_config,
        time_embedding_type=time_embedding_type,
    )
    return model


class RollingShutterDeblur(nn.Module):
    def __init__(
        self,
        image_channel,
        coords_dim,
        events_moment,
        meta_type,
        encoder_name,
        decoder_name,
        inr_depth,
        inr_in_channel,
        inr_mid_channel,
        image_height,
        image_width,
        rs_blur_timestamp,
        gs_sharp_timestamps,
        rs_sharp_timestamps,
        intermediate_visualization,
        dcn_config,
        esl_config,
        time_embedding_type,
    ):
        super().__init__()
        # assert
        assert coords_dim in [1, 3], "coords_dim should be 1 or 3"
        assert image_channel in [1, 3], "image_channel should be 1 or 3"
        assert len(rs_blur_timestamp) == 3, "[start_time, end_time, exposuer_time]"
        # 1. set config
        # 1.1 network setting
        self.meta_type = meta_type
        self.encoder_name = encoder_name
        self.decoder_name = decoder_name
        self.inr_depth = inr_depth
        self.inr_in_channel = inr_in_channel
        self.inr_mid_channel = inr_mid_channel
        # 1.2
        self.events_moment = events_moment
        self.coords_dim = coords_dim
        self.image_height = image_height
        self.image_width = image_width
        self.image_channel = image_channel
        self.rs_blur_timestamp = rs_blur_timestamp
        self.gs_sharp_timestamps = gs_sharp_timestamps
        self.rs_sharp_timestamps = rs_sharp_timestamps
        self.intermediate_visualization = intermediate_visualization
        self.dcn_config = dcn_config
        self.esl_config = esl_config
        self.time_embedding_type = time_embedding_type
        # 2. set model
        self.encoder = self._get_encoder()
        # 3. set coordinate
        self.gs_sharp_coordinates = self._get_gs_sharp_coordinates(coords_dim)
        self.rs_sharp_coordinates = self._get_rs_sharp_coordinates(coords_dim)

        #
        self._info()

    def forward(self, batch):
        # 1. get batch
        # B, C, H, W
        if self.image_channel == 1:
            blur_frame = batch["rolling_blur_frame_gray"]
        else:
            blur_frame = batch["rolling_blur_frame_color"]
        device = blur_frame.device
        gs_sharp_coordinates = [c.to(device) for c in self.gs_sharp_coordinates]
        rs_sharp_coordinates = [c.to(device) for c in self.rs_sharp_coordinates]
        B, C, H, W = blur_frame.shape
        # B, C, H, W
        events = batch["events"]
        # 2. INR generation
        inr = self.encoder(events, blur_frame)

        # For decode part, the input is only a coordinate.
        # 3. decoder for each sharp frame
        # 3.1 decode the gs sharp
        gs_sharp_frame_list = []
        gs_sharp_frame_differential_list = []
        # random select a time stamps
        for i in range(len(self.gs_sharp_timestamps)):
            gs_timestamp = self.gs_sharp_timestamps[i]
            gs_s_coordinate = gs_sharp_coordinates[i]
            if DEBUG:
                # C, H, W
                input_global_sharp_frame_timestamps = batch["global_sharp_frame_timestamps"][0][i]
                if self.coords_dim == 3:
                    xy, t = (gs_s_coordinate[0:2, :, :], gs_s_coordinate[2, :, :])
                else:
                    t = gs_s_coordinate
                info(f" GS [{i}]                           : {gs_timestamp}")
                info(f" input_global_sharp_frame_timestamps: {input_global_sharp_frame_timestamps}")
                assert abs(t.min().item() - gs_timestamp) < 1e-6
                assert abs(t.max().item() - gs_timestamp) < 1e-6
                assert abs(input_global_sharp_frame_timestamps.item() - gs_timestamp) < 1e-4
            gs_s_coordinate = gs_s_coordinate.unsqueeze(0).expand(B, -1, -1, -1)
            gs_s_coordinate = gs_s_coordinate.detach().requires_grad_(True).cuda()
            # for coords inference
            gs_s_frame_t = inr(gs_s_coordinate)
            if self.training:
                gs_s_frame_t_d = torch.autograd.grad(
                    outputs=gs_s_frame_t,
                    inputs=gs_s_coordinate,
                    grad_outputs=torch.ones_like(gs_s_frame_t),
                    create_graph=True,
                    retain_graph=True,
                )[0]
            else:
                gs_s_frame_t_d = torch.zeros_like(gs_s_coordinate).to(gs_s_frame_t.device)
            #
            if self.coords_dim == 3:
                # B, C (dx, dy, dt), H, W -> B, 1 (dt), H, W
                gs_s_frame_t_d = gs_s_frame_t_d[:, 2:3, :, :]

            gs_sharp_frame_list.append(gs_s_frame_t)
            gs_sharp_frame_differential_list.append(gs_s_frame_t_d)
        # (B, C, H, W) N -> N B C H W
        gs_sharp_frame_list = torch.stack(gs_sharp_frame_list)
        # N B C H W -> B N C H W
        batch["global_sharp_pred_frames"] = gs_sharp_frame_list.permute(1, 0, 2, 3, 4)
        # (B, C, H, W) N -> N B C H W
        gs_sharp_frame_differential_list = torch.stack(gs_sharp_frame_differential_list)
        batch["global_sharp_pred_frames_differential"] = gs_sharp_frame_differential_list.permute(1, 0, 2, 3, 4)
        # 3.2 decode the rs sharp
        rs_accumulate = 0
        rolling_sharp_pred_frames = []
        for i in range(len(self.rs_sharp_timestamps)):
            rs_start, rs_end = self.rs_sharp_timestamps[i]
            # rs_s_coordinate: 3, H, W
            rs_s_coordinate = rs_sharp_coordinates[i]
            if DEBUG:
                if self.coords_dim == 3:
                    xy, t = (rs_s_coordinate[0:2, :, :], rs_s_coordinate[2, :, :])
                else:
                    t = rs_s_coordinate
                assert abs(t.min().item() - rs_start) < 1e-6
                assert abs(t.max().item() - rs_end) < 1e-6
                info(f"RS [{i}]: {rs_start} {rs_end}")
            # C, H, W -> B, C, H, W
            rs_s_coordinate = rs_s_coordinate.unsqueeze(0).expand(B, -1, -1, -1)
            rs_s_frame = inr(rs_s_coordinate)
            rolling_sharp_pred_frames.append(rs_s_frame)
            rs_accumulate = rs_accumulate + rs_s_frame
        rs_blur_reconstructed = rs_accumulate / len(self.rs_sharp_timestamps)
        # 1, 1, 256, 256
        if self.intermediate_visualization:
            rolling_sharp_pred_frames = torch.stack(rolling_sharp_pred_frames)
            batch["rolling_sharp_pred_frames"] = rolling_sharp_pred_frames.permute(1, 0, 2, 3, 4)
        batch["rolling_blur_pred_frame"] = rs_blur_reconstructed
        return batch

    def _info(self):
        info(f"{__class__}:")
        info(f"  meta_type: {self.meta_type}")
        info(f"  encoder_name: {self.encoder_name}")
        info(f"  decoder_name: {self.decoder_name}")
        info(f"  inr_in_channel: {self.inr_in_channel}")
        info(f"  inr_mid_channel: {self.inr_mid_channel}")
        info(f"  events_moment: {self.events_moment}")
        info(f"  image_height: {self.image_height}")
        info(f"  image_width: {self.image_width}")
        info(f"  image_channel: {self.image_channel}")
        info(f"  rs_blur_timestamp: {self.rs_blur_timestamp}")
        info(f"  gs_sharp_timestamps: {self.gs_sharp_timestamps}")
        info(f"  rs_sharp_timestamps: {self.rs_sharp_timestamps}")
        info(f"  intermediate_visualization: {self.intermediate_visualization}")
        info(f"  time_embedding_type: {self.time_embedding_type}")

    def _get_gs_sharp_coordinates(self, dim):
        with_position = dim == 3
        gs_sharp_coordinates = []
        for timestamp in self.gs_sharp_timestamps:
            gs_sharp_coordinate = get_t_global_shutter_coordinate(
                timestamp, self.image_height, self.image_width, with_position
            )
            gs_sharp_coordinates.append(gs_sharp_coordinate)
        return gs_sharp_coordinates

    def _get_rs_sharp_coordinates(self, dim):
        with_position = dim == 3
        rs_sharp_coordinates = []
        for timestamp in self.rs_sharp_timestamps:
            t_start, t_end = timestamp
            # dim, h, w
            rs_sharp_coordinate = get_t_rolling_shutter_coordinate(
                t_start, t_end, self.image_height, self.image_width, with_position
            )
            rs_sharp_coordinates.append(rs_sharp_coordinate)
        return rs_sharp_coordinates

    def _get_encoder(self):
        if self.meta_type == "UNetCon1x1WithTheta":
            model = UNetCon1x1WithTheta(
                encoder_type=self.encoder_name,
                decoder_type=self.decoder_name,
                coords_dim=self.coords_dim,
                depth=self.inr_depth,
                meta_in_channels=self.events_moment + self.image_channel,
                meta_out_channels=self.inr_in_channel,
                inr_in_channel=self.inr_in_channel,
                inr_mid_channel=self.inr_mid_channel,
                inr_out_channel=self.image_channel,
                dcn_config=self.dcn_config,
                esl_config=self.esl_config,
                time_embedding_type=self.time_embedding_type,
            )
        else:
            raise NotImplementedError
        return model
