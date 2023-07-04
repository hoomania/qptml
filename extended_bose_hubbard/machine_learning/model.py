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


class CNN1D(nn.Module):

    def __init__(self, lattice_length: int):
        super().__init__()
        self.conv1 = nn.Conv1d(10, 32, kernel_size=3, stride=1)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.3)

        self.conv2 = nn.Conv1d(32, 10, kernel_size=3, stride=1)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.flat = nn.Flatten()

        # nodes = int(32 * (lattice_length-4)/2 * (lattice_length-4)/2)
        # nodes = int(32 * (lattice_length-2) * (lattice_length-2))
        self.fc3 = nn.Linear(10*14, 100)
        self.act3 = nn.ReLU()
        self.drop3 = nn.Dropout(0.5)

        self.fc4 = nn.Linear(100, 3)  # 512

    def forward(self, x):
        # input 10 x L, output 32 x (L-2)
        x = self.conv1(x)
        x = self.act1(x)
        x = self.drop1(x)
        # input 32 x (L-2), output 32 x (L-4) x (L-4)
        x = self.act2(self.conv2(x))
        # input 32 x (L-4) x (L-4), output 32 x (L-4)/2 x (L-4)/2
        x = self.pool2(x)
        # input 32 x (L-4)/2 x (L-4)/2, output 32 * (L-4)/2 * (L-4)/2
        x = self.flat(x)
        # input 8192, output 512
        # print(x)
        x = self.act3(self.fc3(x))
        x = self.drop3(x)
        # input 512, output 2
        x = self.fc4(x)
        # return self.softmax(x)
        return x
