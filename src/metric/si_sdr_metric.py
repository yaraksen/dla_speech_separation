from typing import List

import torch
from torch import Tensor

from torchmetrics.audio import ScaleInvariantSignalDistortionRatio
from src.base.base_metric import BaseMetric

class SiSDRMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.si_sdr = ScaleInvariantSignalDistortionRatio()
    
    def to(self, device):
        self.si_sdr = self.si_sdr.to(device)
        return self

    def __call__(self, pred_short: Tensor, target: Tensor, **kwargs):
        return self.si_sdr(pred_short, target)
