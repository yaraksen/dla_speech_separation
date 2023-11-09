from typing import List

import torch
from torch import Tensor

from src.base.base_metric import BaseMetric

class SNRMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, est: Tensor, target: Tensor, **kwargs):
        alpha = torch.sum(target * est) / torch.square(torch.norm(target))
        return 20 * torch.log10(torch.norm(alpha * target) / (torch.norm(alpha * target - est) + 1e-6) + 1e-6)