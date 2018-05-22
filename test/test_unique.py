from itertools import product

import pytest
from torch_unique import unique

from .utils import dtypes, devices, tensor


@pytest.mark.parametrize('dtype,device', product(dtypes, devices))
def test_unique(dtype, device):
    src = tensor([100, 10, 100, 1, 200, 1, 200, 10], dtype, device)

    out, perm = unique(src)
    assert out.tolist() == [1, 10, 100, 200]
    assert out.tolist() == src[perm].tolist()
