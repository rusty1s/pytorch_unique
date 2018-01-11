import torch
import numpy as np

from .._ext import ffi

def unique(input):
    if input.is_cuda:
        output = input.new(input.size()).copy_(input)
        typename = type(input).__name__.replace('Tensor', '')
        func = getattr(ffi, 'unique_single_cuda_{}'.format(typename))
        func(output)
        return output
    else:
        output, unique_index = np.unique(input.numpy(), return_inverse=True)
        return torch.from_numpy(output), torch.from_numpy(unique_index)

