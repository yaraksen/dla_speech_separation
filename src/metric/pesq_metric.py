from typing import List

import torch
from torch import Tensor, zeros_like
from torch.nn.functional import pad

from torchmetrics.functional.audio.pesq import perceptual_evaluation_speech_quality
from src.base.base_metric import BaseMetric

def to_real_length(t: Tensor, mixed_lens: Tensor) -> Tensor:
    masked = zeros_like(t)
    for row, len in enumerate(mixed_lens):
        masked[row, :len] = t[row, :len]
    return masked

class PESQMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.pesq = PerceptualEvaluationSpeechQuality(fs=16000, mode="wb")
    
    # def to(self, device):
    #     self.pesq = self.pesq.to(device)
    #     return self

    @torch.no_grad()
    def __call__(self, pred_short: Tensor, target: Tensor, mixed_lens, **kwargs):
        target = to_real_length(target, mixed_lens)
        pred_short = to_real_length(pred_short, mixed_lens)

        pad_value = pred_short.shape[-1] - target.shape[-1]
        if pad_value >= 0:
            target = pad(target, (0, pad_value))
        else:
            pred_short = pad(pred_short, (0, -pad_value))

        return perceptual_evaluation_speech_quality(pred_short, target, fs=16000, mode="wb").mean().item()
