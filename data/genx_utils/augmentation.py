from omegaconf import DictConfig
from torchdata.datapipes.iter import IterDataPipe

from data.utils.augmentor import RandomSpatialAugmentorGenX


class RandAugmentIterDataPipe(IterDataPipe):
    """Applies one random spatial augmentation to a whole stream."""

    def __init__(self, source_dp: IterDataPipe, dataset_config: DictConfig):
        super().__init__()
        self.source_dp = source_dp

        resolution_hw = tuple(dataset_config.resolution_hw)
        assert len(resolution_hw) == 2
        if dataset_config.downsample_by_factor_2:
            resolution_hw = tuple(x // 2 for x in resolution_hw)

        self.spatial_augmentor = RandomSpatialAugmentorGenX(
            dataset_hw=resolution_hw,
            automatic_randomization=False,
            augm_config=dataset_config.data_augmentation.stream)

    def __iter__(self):
        self.spatial_augmentor.randomize_augmentation()
        for x in self.source_dp:
            yield self.spatial_augmentor(x)
