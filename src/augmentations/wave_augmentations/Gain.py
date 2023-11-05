import torch_audiomentations
from torch import Tensor, clip

from hw_asr.augmentations.base import AugmentationBase


class Gain(AugmentationBase):
    def __init__(self, *args, **kwargs):
        self._aug = torch_audiomentations.Gain(*args, **kwargs)

    def __call__(self, data: Tensor):
        x = data.unsqueeze(1)
        return clip(self._aug(x).squeeze(1), min=-1.0, max=1.0)
