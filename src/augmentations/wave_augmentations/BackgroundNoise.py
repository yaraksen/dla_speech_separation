import torch_audiomentations
from torch import Tensor

from hw_asr.augmentations.base import AugmentationBase
from hw_asr.augmentations.utils import download_asset


class BackgroundNoise(AugmentationBase):
    def __init__(self, *args, **kwargs):
        SAMPLE_NOISE = download_asset("tutorial-assets/Lab41-SRI-VOiCES-rm1-babb-mc01-stu-clo-8000hz.wav")
        self._aug = torch_audiomentations.AddBackgroundNoise(background_paths=SAMPLE_NOISE, sample_rate=16000, *args, **kwargs)

    def __call__(self, data: Tensor):
        x = data.unsqueeze(1)
        return self._aug(x).squeeze(1)
