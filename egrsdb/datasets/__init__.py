#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from egrsdb.datasets.davis_rs_event import get_dre_dataset
from egrsdb.datasets.evunroll_real_dataset import get_evunroll_real_dataset
from egrsdb.datasets.evunroll_simulated_dataset import get_evunroll_simulated_dataset_with_config
from egrsdb.datasets.fastec_rsb import get_fastec_rolling_shutter_blur_dataset
from egrsdb.datasets.gev_rsb import get_gev_rolling_shutter_blur_dataset


def get_dataset(config):
    if config.NAME == "gev-rolling-shutter-blur":
        return get_gev_rolling_shutter_blur_dataset(
            root=config.root,
            blur_accumulate=config.blur_accumulate,
            gs_sharp_frame_count=config.gs_sharp_frame_count,
            events_moment=config.events_moment,
            center_cropped_height=config.center_cropped_height,
            random_cropped_width=config.random_cropped_width,
            is_color=config.is_color,
            gs_sharp_start_index=config.gs_sharp_start_index,
            gs_sharp_end_index=config.gs_sharp_end_index,
            calculate_in_linear_domain=config.calculate_in_linear_domain,
            event_for_gs_frame_buffer=config.event_for_gs_frame_buffer,
            correct_offset=config.correct_offset,
        )
    elif config.NAME == "fastec-rolling-shutter-blur":
        return get_fastec_rolling_shutter_blur_dataset(
            root=config.root,
            blur_accumulate=config.blur_accumulate,
            gs_sharp_frame_count=config.gs_sharp_frame_count,
            events_moment=config.events_moment,
            center_cropped_height=config.center_cropped_height,
            random_cropped_width=config.random_cropped_width,
            is_color=config.is_color,
            gs_sharp_start_index=config.gs_sharp_start_index,
            gs_sharp_end_index=config.gs_sharp_end_index,
            calculate_in_linear_domain=config.calculate_in_linear_domain,
            event_for_gs_frame_buffer=config.event_for_gs_frame_buffer,
            correct_offset=config.correct_offset,
        )
    elif config.NAME == "evunroll-real":
        test = get_evunroll_real_dataset(
            fps=20.79,
            data_root=config.data_root,
            moments=config.events_moment,
            is_color=config.is_color,
        )
        return test, test
    elif config.NAME == "evunroll-simulated":
        return get_evunroll_simulated_dataset_with_config(config)
    elif config.NAME == "dre-real":
        return get_dre_dataset(config)
    else:
        raise ValueError(f"Unknown dataset: {config.NAME}")
