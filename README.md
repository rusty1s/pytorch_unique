[pypi-image]: https://badge.fury.io/py/torch-unique.svg
[pypi-url]: https://pypi.python.org/pypi/torch-unique
[build-image]: https://travis-ci.org/rusty1s/pytorch_unique.svg?branch=master
[build-url]: https://travis-ci.org/rusty1s/pytorch_unique
[coverage-image]: https://codecov.io/gh/rusty1s/pytorch_unique/branch/master/graph/badge.svg
[coverage-url]: https://codecov.io/github/rusty1s/pytorch_unique?branch=master

# PyTorch Unique

[![PyPI Version][pypi-image]][pypi-url]
[![Build Status][build-image]][build-url]
[![Code Coverage][coverage-image]][coverage-url]

--------------------------------------------------------------------------------

This package consists of a small extension library of highly optimized unique operations for the use in [PyTorch](http://pytorch.org/), which are missing in the main package.
The package consists of the following operations:

* **[Unique](#unique)**
* **[Unique by Key](#unique-by-key)**

All included operations work on varying data types and are implemented both for CPU and GPU.

## Installation

Check that `nvcc` is accessible from terminal, e.g. `nvcc --version`.
If not, add cuda (`/usr/local/cuda/bin`) to your `$PATH`.
Then run:

```
pip install cffi torch-unique
```

## Unique

Returns the sorted unique elements of an one-dimensional tensor.

```py
import torch
from torch_unique import unique

src = torch.tensor([100, 10, 100, 1, 1000, 1, 1000, 1])
out = unique(src)
```

```
print(out)
tensor([ 1,  10,  100,  1000])
```

## Unique by Key

Returns the sorted unique elements of the one-dimensional tensor `key` as first return value.
In addition, `value` is filtered by the unique indices of `key`.

```py
import torch
from torch_unique import unique_by_key

key = torch.tensor([100, 10, 100, 1, 1000, 1, 1000, 1])
value = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])
key_out, value_out = unique_by_key(key, value)
```

```
print(key_out)
tensor([ 1,  10,  100,  1000])

print(value_out)
tensor([ 4,  2,  1,  5])
```

## Running tests

```
python setup.py test
```
