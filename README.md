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

**PyTorch 0.4.1 now supports [`unique`](https://pytorch.org/docs/stable/torch.html#torch.unique) both for CPU and GPU.
Therefore, this package is no longer needed and will not be updated.
In contrast to this package, PyTorch's version does not return an index array.
However, you can easily generate it by using the following code:**

```python
import torch

unique, inverse = torch.unique(x, sorted=True, return_inverse=True)
perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
perm = inverse.new_empty(unique.size(0)).scatter_(0, inverse, perm)
```

--------------------------------------------------------------------------------

This package consists of a small extension library of a highly optimized `unique` operation for the use in [PyTorch](http://pytorch.org/), which is missing in the main package.
The operation works on varying data types and is implemented both for CPU and GPU.

## Installation

Ensure that PyTorch 0.4.0 is installed and verify that `cuda/bin` and `cuda/install` are in your `$PATH` and `$CPATH` respectively, *e.g.*:

```
$ python -c "import torch; print(torch.__version__)"
>>> 0.4.0

$ echo $PATH
>>> /usr/local/cuda/bin:...

$ echo $CPATH
>>> /usr/local/cuda/install:...
```

Then run:

```
pip install torch-scatter torch-unique
```

If you are running into any installation problems, please create an [issue](https://github.com/rusty1s/pytorch_unique/issues).

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
