import torch
from torch_unique import unique, unique_by_key
import time

# print(ffi.__dict__)

# f

input = torch.arange(0, 100000000)

t = time.process_time()
input = unique(input)
t = time.process_time() - t 
print(t)
input = input.cuda()
t = time.process_time()
input = unique(input)
t = time.process_time() - t 
print(t)
# ffi.unique_single_cuda_Long(input)
# print(input)

# ffi.unique_byKey_cuda_Long(input, value)
# print(input)
# print(value)

# output, index = unique(input)
# print(input)
# print(output)
