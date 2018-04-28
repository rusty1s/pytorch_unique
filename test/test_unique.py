from itertools import product

import pytest
from torch_unique import unique

from .utils import dtypes, devices, tensor


@pytest.mark.parametrize('dtype,device', product(dtypes, devices))
def test_unique(dtype, device):
    src = tensor([100, 10, 100, 1, 1000, 1, 1000, 10], dtype, device)
    assert unique(src).tolist() == [1, 10, 100, 1000]
