import torch
from torch_unique import unique

# print(ffi.__dict__)

# f

input = torch.cuda.LongTensor([1, 1, 1, 1, 1000, 1, 1000, 100])
value = torch.cuda.LongTensor([1, 2, 2, 1, 3, 1, 1000, 100])
output = unique(input)
# ffi.unique_single_cuda_Long(input)
# print(input)

# ffi.unique_byKey_cuda_Long(input, value)
# print(input)
# print(value)

# output, index = unique(input)
# print(input)
# print(output)
