import torch
import numpy as np

if torch.cuda.is_available():
    import unique_cuda


def unique(src):
    """Returns the sorted unique scalar elements of the input tensor as an
    one-dimensional tensor.

    A tuple of :obj:`(unique_tensor, unique_indices)` is returned, where the
    :obj:`unique_indices` are the indices of the elements in the original input
    tensor. Note that :obj:`unique_indices` is not guaranteed to be stable on
    GPU.

    Args:
        src (Tensor): The input tensor.

    :rtype: (:class:`Tensor`, :class:`LongTensor`)
    """

    if src.is_cuda:
        out, perm = unique_cuda.unique(src)
    else:
        out, perm = np.unique(src.numpy(), return_index=True)
        out, perm = torch.from_numpy(out), torch.from_numpy(perm)
    return out, perm
