import unittest
from hw_asr.base.base_text_encoder import BaseTextEncoder

from hw_asr.text_encoder.ctc_char_text_encoder import CTCCharTextEncoder
from hw_asr.metric.utils import calc_cer, calc_wer
import pickle
from torch import tensor

class TestTextEncoder(unittest.TestCase):
    def test_ctc_decode(self):
        text_encoder = CTCCharTextEncoder()
        text = "i^^ ^w^i^sss^hhh^   i ^^^s^t^aaaar^teee^d " \
               "dddddd^oooo^in^g tttttttth^iiiis h^^^^^^^^w^ e^a^r^li^er"
        true_text = "i wish i started doing this hw earlier"
        inds = [text_encoder.char2ind[c] for c in text]
        decoded_text = text_encoder.ctc_decode(inds)
        self.assertIn(decoded_text, true_text)

    def test_beam_search(self):
        text_encoder = CTCCharTextEncoder()

        # Load precomputed CTC output
        with open('aux/lj_batch.pickle', 'rb') as f:
            batch = pickle.load(f)

        # log probabilities of softmax layers [batch_size, T, vocab_size]
        log_probs = batch["log_probs"]
        log_probs_length = tensor([log_probs[i].shape[0] for i in range(batch["log_probs"].shape[0])])
        true_texts = batch["text"]

        bs_results = []
        for i in range(log_probs.shape[0]):
            bs_results.append(text_encoder.ctc_beam_search(log_probs[i].exp().numpy(), log_probs_length[i], 3))

        lmbs_results = text_encoder.ctc_lm_beam_search(log_probs, log_probs_length, alpha=0.5, beta=1.5)
        lmbs_results = [BaseTextEncoder.normalize_text(pred) for pred in lmbs_results]

        for i in range(len(true_texts)):
            true_text = true_texts[i]
            argmax_text = text_encoder.ctc_decode(log_probs[i].numpy().argmax(-1))
            print(f"True: '{true_text}'")
            print(f"Argmax: '{argmax_text}' --- (CER: {calc_cer(true_text, argmax_text):.3f})")
            print(f"BS: '{bs_results[i][0].text}' --- (CER: {calc_cer(true_text, bs_results[i][0].text):.3f})")
            print(f"LMBS: '{lmbs_results[i]}' --- (CER: {calc_cer(true_text, lmbs_results[i]):.3f})")
            print('-' * 100)

