import torch

m = torch.nn.Conv1d(16, 1, 3, stride=2)
data = torch.randn(1, 16, 5)
output = m(data)

print(data)
