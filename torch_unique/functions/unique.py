import torch
import numpy as np

from .._ext import ffi

def unique(input):
    if input.is_cuda:
        output, sort_index = input.sort(dim=0)

        typename = type(input).__name__.replace('Tensor', '')
        func = getattr(ffi, 'unique_cuda_{}'.format(typename))
        unique_index = torch.cuda.LongTensor(input.size())
        func(index, output)

        index = input.new(input.size()).long()
        return output, index
    else:
        output, unique_index = np.unique(input.numpy(), return_inverse=True)
        return torch.from_numpy(output), torch.from_numpy(unique_index)

