from torch import Tensor, zeros_like
from torch.nn import CrossEntropyLoss
from torch.nn.functional import pad
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio
from torch.nn import Module
import torch


def zero_mean(t: torch.Tensor):
    return t - torch.mean(t, dim=-1, keepdim=True)

def si_sdr(est: Tensor, target: Tensor, **kwargs):
    alpha = (torch.sum(target * est, dim=-1) / torch.square(torch.linalg.norm(target, dim=-1))).unsqueeze(1)
    return 20 * torch.log10(torch.linalg.norm(alpha * target, dim=-1) / (torch.linalg.norm(alpha * target - est, dim=-1) + 1e-6) + 1e-6)

def to_real_length(t: Tensor, mixed_lens: Tensor) -> Tensor:
    masked = zeros_like(t)
    for row, len in enumerate(mixed_lens):
        masked[row, :len] = t[row, :len]
    return masked

class SpexLoss(Module):
    def __init__(self, alpha, beta, gamma):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.ce = CrossEntropyLoss()

    def forward(self, is_train: bool, **batch) -> Tensor:
        num_samples = batch["target"].shape[0]
        target = to_real_length(batch["target"], batch["mixed_lens"])

        pad_value = batch["pred_short"].shape[-1] - batch["target"].shape[-1]
        assert pad_value >= 0, f"pad_value {pad_value} < 0"
        target = pad(target, (0, pad_value))

        short_si_sdr = si_sdr(to_real_length(batch["pred_short"], batch["mixed_lens"]),
                                   target).sum()
        mid_si_sdr = si_sdr(to_real_length(batch["pred_mid"], batch["mixed_lens"]),
                                 target).sum()
        long_si_sdr = si_sdr(to_real_length(batch["pred_long"], batch["mixed_lens"]),
                                  target).sum()

        si_sdr_loss = -((1 - self.alpha - self.beta) * short_si_sdr + self.alpha * mid_si_sdr + self.beta * long_si_sdr) / num_samples
        if not is_train:
            return si_sdr_loss

        ce_loss = self.ce(batch["speakers_logits"], batch["speakers_ids"])
        return si_sdr_loss + self.gamma * ce_loss
