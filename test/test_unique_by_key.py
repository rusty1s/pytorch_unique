from itertools import product

import pytest
from torch_unique import unique_by_key

from .utils import dtypes, devices, tensor


@pytest.mark.parametrize('dtype,device', product(dtypes, devices))
def test_unique_by_key(dtype, device):
    key = tensor([100, 10, 100, 1, 1000, 1, 1000, 10], dtype, device)
    value = tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype, device)

    out = unique_by_key(key, value)
    assert out[0].tolist() == [1, 10, 100, 1000]
    assert out[1].tolist() == [4, 2, 1, 5]
