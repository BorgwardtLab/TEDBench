import torch
import numpy as np
from typing import List


def pad_structures(
    items, constant_value=0, dtype=None, truncation_length=600, pad_length=None
):
    """Reference to TAPE https://github.com/songlab-cal/tape/blob/6d345c2b2bbf52cd32cf179325c222afd92aec7e/tape/datasets.py#L37"""
    batch_size = len(items)
    if isinstance(items[0], List):
        items = [torch.tensor(x) for x in items]
    if pad_length is None:
        shape = [batch_size] + np.max([x.shape for x in items], 0).tolist()
    else:
        shape = [batch_size] + [pad_length]
    if shape[1] > truncation_length:
        shape[1] = truncation_length

    if dtype is None:
        dtype = items[0].dtype

    if isinstance(items[0], np.ndarray):
        array = np.full(shape, constant_value, dtype=dtype)
    else:
        array = torch.full(shape, constant_value, dtype=dtype)

    for arr, x in zip(array, items):
        arrslice = tuple(slice(dim) for dim in x.shape)
        arr[arrslice] = x[:truncation_length]

    return array
