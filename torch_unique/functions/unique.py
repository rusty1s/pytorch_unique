import torch

from .._ext import ffi

def unique(input):
    typename = type(input).__name__.replace('Tensor', '')
    cuda = 'cuda_' if input.is_cuda else ''
    func = getattr(ffi, 'unique_{}{}'.format(cuda, typename))
    index = input.new(input.size()).long()
    output = func(index, input)
    return output, index

