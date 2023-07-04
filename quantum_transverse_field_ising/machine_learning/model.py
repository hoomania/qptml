import torch.nn as nn


class FC(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_layer: int = 100):
        super(FC, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_layer)
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_layer, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.sigmoid(out)
        out = self.fc2(out)
        return self.softmax(out)
