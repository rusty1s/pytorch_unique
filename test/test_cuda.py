import torch
from torch_unique import unique

input = torch.cuda.IntTensor([1, 1, 1, 1, 1000, 1000, 1000, 100])

output, index = unique(input)
print(input)
# print(output)
