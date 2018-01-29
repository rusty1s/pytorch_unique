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
