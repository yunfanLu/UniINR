from egrsdb.visualize.rolling_shutter_visualization import VisualizationRollingShutter


def get_visulization(config):
    if config.NAME == "rs-vis":
        return VisualizationRollingShutter(config)
    else:
        raise NotImplementedError(f"Visualization {config.NAME} is not implemented.")
