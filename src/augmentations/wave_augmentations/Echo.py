import torch_audiomentations
from torch import Tensor

from hw_asr.augmentations.base import AugmentationBase
from hw_asr.augmentations.utils import download_asset


class Echo(AugmentationBase):
    def __init__(self, *args, **kwargs):
        SAMPLE_RIR = download_asset("tutorial-assets/Lab41-SRI-VOiCES-rm1-impulse-mc01-stu-clo-8000hz.wav")
        self._aug = torch_audiomentations.ApplyImpulseResponse(ir_paths=SAMPLE_RIR,
                                                               sample_rate=16000,
                                                               compensate_for_propagation_delay=True,
                                                               *args, **kwargs)

    def __call__(self, data: Tensor):
        x = data.unsqueeze(1)
        return self._aug(x).squeeze(1)
