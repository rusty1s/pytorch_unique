import pytest
import torch
import torch_unique

from .utils import tensors, Tensor


@pytest.mark.parametrize('tensor', tensors)
def test_unique_by_key_cpu(tensor):
    key = Tensor(tensor, [100, 10, 100, 1, 1000, 1, 1000, 10])
    value = Tensor(tensor, [1, 2, 3, 4, 5, 6, 7, 8])
    expected_key = Tensor(tensor, [1, 10, 100, 1000])
    expected_value = Tensor(tensor, [4, 2, 1, 5])

    output_key, output_value = torch_unique.unique_by_key(key, value)
    assert output_key.tolist() == expected_key.tolist()
    assert output_value.tolist() == expected_value.tolist()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='no CUDA')
@pytest.mark.parametrize('tensor', tensors)
def test_unique_by_key_gpu(tensor):  # pragma: no cover
    key = Tensor(tensor, [100, 10, 100, 1, 1000, 1, 1000, 10]).cuda()
    value = Tensor(tensor, [1, 2, 3, 4, 5, 6, 7, 8]).cuda()
    expected_key = Tensor(tensor, [1, 10, 100, 1000])
    expected_value = Tensor(tensor, [4, 2, 1, 5])

    output_key, output_value = torch_unique.unique_by_key(key, value)
    assert output_key.cpu().tolist() == expected_key.tolist()
    assert output_value.cpu().tolist() == expected_value.tolist()
