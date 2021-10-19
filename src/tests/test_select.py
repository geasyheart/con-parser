# -*- coding: utf8 -*-
#
import torch

a = torch.arange(24).reshape(2, 3, 4)
print(a)

index = torch.tensor([
    [0, 1, 2],
    [0, 1, 0]
])

result = torch.gather(a, dim=1, index=index.unsqueeze(-1).expand(-1, -1, a.size(-1)))
print(result)
