import torch
import numpy as np

from .._ext import ffi

def unique(input):
    assert input.dim() == 1, 'Input tensor must be 1-dimensional'

    if input.is_cuda:
        output = input.new(input.size()).copy_(input)
        typename = type(input).__name__.replace('Tensor', '')
        func = getattr(ffi, 'unique_cuda_{}'.format(typename))
        func(output)
        return output
    else:
        input = input.numpy()
        input = np.unique(input)
        return torch.from_numpy(input)


def unique_by_key(key, value):
    assert key.dim() == 1, 'Key tensor must be 1-dimensional'
    assert key.dim() == value.dim(), ('Key tensor must have same dimensions as '
                                      'value tensor')
    assert key.numel() == value.numel(), ('Key tensor must have same size as '
                                          'value tensor')

    if key.is_cuda:
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
