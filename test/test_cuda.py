import torch
from torch_unique import unique, unique_by_key
import time

# print(ffi.__dict__)

# f

key = torch.cuda.LongTensor([1, 1, 1, 1, 1000, 1, 1000, 100])
value = torch.cuda.LongTensor([1, 2, 2, 1, 3, 1, 1000, 100])
# input = torch.arange(0, 100000000).cuda()

t = time.process_time()
key, value = unique_by_key(key, value)
t = time.process_time() - t 
print(t)
print(key)
print(value)
# ffi.unique_single_cuda_Long(input)
# print(input)

# ffi.unique_byKey_cuda_Long(input, value)
# print(input)
# print(value)

# output, index = unique(input)
# print(input)
# print(output)
