from egrsdb.models.rsdb import get_rolling_shutter_deblur_model


def get_model(config):
    if config.NAME == "rsdb":
        return get_rolling_shutter_deblur_model(
            image_channel=config.image_channel,
            coords_dim=config.coords_dim,
            events_moment=config.events_moment,
            meta_type=config.meta_type,
            encoder_name=config.encoder_name,
            decoder_name=config.decoder_name,
            inr_depth=config.inr_depth,
            inr_in_channel=config.inr_in_channel,
            inr_mid_channel=config.inr_mid_channel,
            image_height=config.image_height,
            image_width=config.image_width,
            rs_blur_timestamp=config.rs_blur_timestamp,
            gs_sharp_count=config.gs_sharp_count,
            rs_integral=config.rs_integral,
            intermediate_visualization=config.intermediate_visualization,
            dcn_config=config.dcn_config,
            esl_config=config.esl_config,
            correct_offset=config.correct_offset,
            time_embedding_type=config.time_embedding_type,
        )
    else:
        raise ValueError(f"Model {config.NAME} is not supported.")
