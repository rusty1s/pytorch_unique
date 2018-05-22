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

This package consists of a small extension library of a highly optimized `unique` operation for the use in [PyTorch](http://pytorch.org/), which is missing in the main package.
The operation works on varying data types and is implemented both for CPU and GPU.

## Installation

```
pip install torch-unique
```

## Usage

```
torch_unique.unique(src) -> (Tensor, LongTensor)
```

Returns the sorted unique scalar elements of the input tensor as an one-dimensional tensor.

A tuple of `(unique_tensor, unique_indices)` is returned, where the `unique_indices` are the indices of the elements in the original input tensor. Note that `unique_indices` is not guaranteed to be stable on GPU.

### Parameters

* **src** *(Tensor)* - The input tensor.

### Returns

* **out** *(Tensor)* - The unique elements from `src` as an one-dimensional tensor.
* **perm** *(LongTensor)* - The unique indices from `src` as an one-dimensional tensor.

### Example

```py
import torch
from torch_unique import unique

src = torch.tensor([100, 10, 100, 1, 1000, 1, 1000, 1])
out, perm = unique(src)
```

```
print(out)
tensor([    1,    10,   100,  1000])
print(perm)
tensor([ 3,  1,  0,  4])
```

## Running tests

```
python setup.py test
```
