import torch
import numpy as np

from .._ext import ffi


def unique_by_key(key, value):
    """Returns the sorted unique elements of the one-dimensional tensor
    :attr:`key` as first return value. In addition, :attr:`value` is filtered
    by the unique indices of :attr:`key`.

    Args:
        key (Tensor): The key source tensor
        value (Tensor): The value source tensor

    :rtype: (:class:`Tensor`, :class:`Tensor`)
    """

    assert key.dim() == 1, 'Key tensor must be 1-dimensional'
    assert key.dim() == value.dim(), (
        'Key tensor must have same dimensions as '
        'value tensor')
    assert key.numel() == value.numel(), ('Key tensor must have same size as '
                                          'value tensor')

    if key.is_cuda:  # pragma: no cover
        key = key.new(key.size()).copy_(key)
        value = value.new(value.size()).copy_(value)
        typename = type(key).__name__.replace('Tensor', '')
        func = getattr(ffi, 'uniqueByKey_cuda_{}'.format(typename))
        func(key, value)
        return key, value
    else:
        key, value = key.numpy(), value.numpy()
        key, index = np.unique(key, return_index=True)
        value = value[index]
        return torch.from_numpy(key), torch.from_numpy(value)
