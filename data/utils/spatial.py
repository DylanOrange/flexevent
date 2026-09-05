def get_dataloading_hw(dataset_config):
    height, width = (int(value) for value in dataset_config.resolution_hw)
    if dataset_config.downsample_by_factor_2:
        height, width = height // 2, width // 2
    return height, width
