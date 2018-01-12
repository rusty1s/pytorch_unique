# PyTorch Unique

--------------------------------------------------------------------------------

This package consists of a small extension library of highly optimised unique operations for the use in [PyTorch](http://pytorch.org/), which are missing in the main package.
The package consists of the following operations:

* `unique`
* `unique_by_key`

All included operations work on varying data types and are implemented both for CPU and GPU.

## Installation

Check that `nvcc` is accessible from terminal, e.g. `nvcc --version`.
If not, add cuda (`/usr/local/cuda/bin`) to your `$PATH`.
Then run:

```
pip install cffi
python setup.py install
```

## Example

```py
import torch
from torch_unique import unique

input = torch.Tensor([2, 0, 1, 4, 3, 0, 2, 1, 3, 4])
output = unique(output)
```

```
print(output)
 0  1  2  3  4
[torch.FloatTensor of size 5]
```

## Running tests

```
python setup.py test
```
