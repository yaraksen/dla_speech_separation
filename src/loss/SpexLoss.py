from torch import Tensor, zeros_like
from torch.nn import CrossEntropyLoss
from torch.nn.functional import pad
from torchmetrics import ScaleInvariantSignalDistortionRatio
from torch.nn import Module


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
        self.si_sdr = ScaleInvariantSignalDistortionRatio(zero_mean=True)
        self.ce = CrossEntropyLoss()

    def forward(self, **batch) -> Tensor:
        num_samples = batch["target"].shape[0]
        target = to_real_length(batch["target"], batch["mixed_lens"])

        pad_value = batch["pred_short"].shape[-1] - batch["target"].shape[-1]
        assert pad_value >= 0, "pad value cannot be less 0"
        target = pad(target, (0, pad_value))

        # getting not mean, but sum
        short_si_sdr = self.si_sdr(to_real_length(batch["pred_short"], batch["mixed_lens"]),
                                   target) * num_samples
        mid_si_sdr = self.si_sdr(to_real_length(batch["pred_mid"], batch["mixed_lens"]),
                                 target) * num_samples
        long_si_sdr = self.si_sdr(to_real_length(batch["pred_long"], batch["mixed_lens"]),
                                  target) * num_samples

        si_sdr_loss = -((1 - self.alpha - self.beta) * short_si_sdr + self.alpha * mid_si_sdr + self.beta * long_si_sdr) / num_samples
        ce_loss = self.ce(batch["speakers_logits"], batch["speakers_ids"])
        return si_sdr_loss + self.gamma * ce_loss
