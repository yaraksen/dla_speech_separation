import logging
from typing import List
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import pad
from torch import tensor

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """

    result_batch = {}
    result_batch["mixed"] = pad_sequence([rec["mixed"].squeeze(0) for rec in dataset_items], True, 0)
    result_batch["ref"] = pad_sequence([rec["ref"].squeeze(0) for rec in dataset_items], True, 0)
    result_batch["target"] = pad_sequence([rec["target"].squeeze(0) for rec in dataset_items], True, 0)

    result_batch["target"] = pad(result_batch["target"], (0, result_batch["mixed"].size(-1) - result_batch["target"].size(-1)))

    result_batch["mixed_lens"] = tensor([rec["mixed"].size(-1) for rec in dataset_items])
    result_batch["ref_lens"] = tensor([rec["ref"].size(-1) for rec in dataset_items])
    result_batch["speakers_ids"] = tensor([rec["speakers_ids"] for rec in dataset_items])
    return result_batch