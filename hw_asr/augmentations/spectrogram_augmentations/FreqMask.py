import torch_audiomentations
from torch import Tensor

from hw_asr.augmentations.base import AugmentationBase
from hw_asr.augmentations.random_apply import RandomApply
from torchaudio.transforms import FrequencyMasking


class FreqMask(AugmentationBase):
    def __init__(self, p: float, *args, **kwargs):
        self._aug = FrequencyMasking(*args, **kwargs)
        self.p = p

    def __call__(self, data: Tensor):
        x = data.unsqueeze(1)
        return RandomApply(self._aug, p=self.p)(x).squeeze(1)
