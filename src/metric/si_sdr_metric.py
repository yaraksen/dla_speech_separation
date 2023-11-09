from typing import List

import torch
from torch import Tensor, zeros_like
from torch.nn.functional import pad

from torchmetrics.audio import ScaleInvariantSignalDistortionRatio
from src.base.base_metric import BaseMetric

def to_real_length(t: Tensor, mixed_lens: Tensor) -> Tensor:
    masked = zeros_like(t)
    for row, len in enumerate(mixed_lens):
        masked[row, :len] = t[row, :len]
    return masked

class SiSDRMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.si_sdr = ScaleInvariantSignalDistortionRatio()
    
    def to(self, device):
        self.si_sdr = self.si_sdr.to(device)
        return self

    def __call__(self, pred_short: Tensor, target: Tensor, mixed_lens, **kwargs):
        target = to_real_length(target, mixed_lens)
        pred_short = to_real_length(pred_short, mixed_lens)

        pad_value = pred_short.shape[-1] - target.shape[-1]
        if pad_value >= 0:
            target = pad(target, (0, pad_value))
        else:
            pred_short = pad(pred_short, (0, -pad_value))

        return self.si_sdr(pred_short, target)
