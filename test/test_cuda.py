import torch
from torch_unique import unique

input = torch.cuda.LongTensor([1, 10, 100, 10, 1000, 1, 1000, 100])

output, index = unique(input)
print(input)
print(index)
# print(output)
