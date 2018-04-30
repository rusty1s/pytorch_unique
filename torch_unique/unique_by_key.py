import torch
import numpy as np

from .utils.ffi import get_func


def unique_by_key(key, value):
    """Returns the sorted unique elements of an one-dimensional tensor
    :attr:`key` as first return value. In addition, :attr:`value` is filtered
    by the unique indices of :attr:`key`.

    Args:
        key (Tensor): The key source tensor.
        value (Tensor): The value source tensor.

    :rtype: (:class:`Tensor`, :class:`Tensor`)
    """

    if key.is_cuda:  # pragma: no cover
        key, value = key.clone(), value.clone()
        get_func('uniqueByKey', key)(key, value)
        return key, value
    else:
        key, index = np.unique(key, return_index=True)
        key = torch.from_numpy(key)
        value = value[index]
        return key, value
