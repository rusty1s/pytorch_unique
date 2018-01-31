import pytest
import torch
from torch_unique import unique

from .utils import tensors, Tensor


@pytest.mark.parametrize('tensor', tensors)
def test_unique_cpu(tensor):
    input = Tensor(tensor, [100, 10, 100, 1, 1000, 1, 1000, 10])
    expected = Tensor(tensor, [1, 10, 100, 1000])

    output = unique(input)
    assert output.tolist() == expected.tolist()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='no CUDA')
@pytest.mark.parametrize('tensor', tensors)
def test_unique_gpu(tensor):  # pragma: no cover
    input = Tensor(tensor, [100, 10, 100, 1, 1000, 1, 1000, 10]).cuda()
    expected = Tensor(tensor, [1, 10, 100, 1000])

    output = unique(input)
    assert output.cpu().tolist() == expected.tolist()
