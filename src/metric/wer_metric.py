from typing import List

import torch
from torch import Tensor

from hw_asr.base.base_metric import BaseMetric
from hw_asr.base.base_text_encoder import BaseTextEncoder
from hw_asr.metric.utils import calc_wer, calc_cer


class ArgmaxWERMetric(BaseMetric):
    def __init__(self, text_encoder: BaseTextEncoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder

    def __call__(self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs):
        wers = []
        predictions = torch.argmax(log_probs.cpu(), dim=-1).numpy()
        lengths = log_probs_length.detach().numpy()
        for log_prob_vec, length, target_text in zip(predictions, lengths, text):
            target_text = BaseTextEncoder.normalize_text(target_text)
            if hasattr(self.text_encoder, "ctc_decode"):
                pred_text = self.text_encoder.ctc_decode(log_prob_vec[:length])
            else:
                pred_text = self.text_encoder.decode(log_prob_vec[:length])
            wers.append(calc_wer(target_text, pred_text))
        return sum(wers) / len(wers)


class BeamSearchWERMetric(BaseMetric):
    def __init__(self, text_encoder: BaseTextEncoder, beam_size: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder
        self.beam_size = beam_size

    def __call__(self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs):
        wers = []
        predictions = log_probs.detach().cpu().numpy().exp()
        lengths = log_probs_length.detach().cpu().numpy()

        for prob_vec, length, target_text in zip(predictions, lengths, text):
            target_text = BaseTextEncoder.normalize_text(target_text)
            if hasattr(self.text_encoder, "ctc_beam_search"):
                pred_text = self.text_encoder.ctc_beam_search(prob_vec, length, self.beam_size)[0].text
            else:
                pred_text = self.text_encoder.decode(prob_vec[:length])
            wers.append(calc_wer(target_text, pred_text))
        return sum(wers) / len(wers)


class LMBeamSearchWERMetric(BaseMetric):
    def __init__(self, text_encoder: BaseTextEncoder, alpha: float, beta: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder
        self.alpha = alpha
        self.beta = beta

    def __call__(self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs):
        text = [BaseTextEncoder.normalize_text(t) for t in text]

        pred_texts = self.text_encoder.ctc_lm_beam_search(log_probs, log_probs_length, self.alpha, self.beta)
        pred_texts = [BaseTextEncoder.normalize_text(pred) for pred in pred_texts]

        wer = 0.0
        for target_text, pred_text in zip(text, pred_texts):
            wer += calc_wer(target_text, pred_text)

        return wer / len(pred_texts)
    

# for speed increase in test.py
class LMBeamSearchAllMetric(BaseMetric):
    def __init__(self, text_encoder: BaseTextEncoder, alpha: float, beta: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder
        self.alpha = alpha
        self.beta = beta

    def __call__(self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs):
        text = [BaseTextEncoder.normalize_text(t) for t in text]

        pred_texts = self.text_encoder.ctc_lm_beam_search(log_probs, log_probs_length, self.alpha, self.beta)
        pred_texts = [BaseTextEncoder.normalize_text(pred) for pred in pred_texts]

        if torch.rand(1) >= 0.8:
            for i in range(10):
                print(f"True: '{text[i]}'")
                print(f"LMBS: '{pred_texts[i]}'")
                print('-' * 100)

        wer, cer = 0.0, 0.0
        for target_text, pred_text in zip(text, pred_texts):
            wer += calc_wer(target_text, pred_text)
            cer += calc_cer(target_text, pred_text)

        return cer / len(pred_texts), wer / len(pred_texts)

class BeamSearchAllMetric(BaseMetric):
    def __init__(self, text_encoder: BaseTextEncoder, beam_size: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder
        self.beam_size = beam_size

    def __call__(self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs):
        wers, cers = [], []
        predictions = log_probs.detach().cpu().exp().numpy()
        lengths = log_probs_length.detach().cpu().numpy()

        for prob_vec, length, target_text in zip(predictions, lengths, text):
            target_text = BaseTextEncoder.normalize_text(target_text)
            if hasattr(self.text_encoder, "ctc_beam_search"):
                pred_text = self.text_encoder.ctc_beam_search(prob_vec, length, self.beam_size)[0].text
            else:
                pred_text = self.text_encoder.decode(prob_vec[:length])
            wers.append(calc_wer(target_text, pred_text))
            cers.append(calc_cer(target_text, pred_text))
        return sum(cers) / len(cers), sum(wers) / len(wers)
