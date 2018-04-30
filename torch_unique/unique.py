import torch
import numpy as np

from .utils.ffi import get_func


def unique(src):
    """Returns the sorted unique elements of an one-dimensional tensor.

    Args:
        src (Tensor): The source tensor.

    :rtype: :class:`Tensor`
    """

    if src.is_cuda:  # pragma: no cover
        out = src.clone()
        get_func('unique', out)(out)
        return out
    else:
        return torch.from_numpy(np.unique(src))
