import numpy as np
import torch
from torch.utils.data import Dataset


class DatasetMaker(Dataset):

    def __init__(self, path: str, row_length: int):
        xy = np.loadtxt(path, delimiter=',', dtype=np.float32, skiprows=1)
        self.n_samples = xy.shape[0]

        self.x_data = torch.from_numpy(xy[:, :row_length])
        self.y_data = torch.nn.functional.one_hot(torch.LongTensor(xy[:, -1])).type(torch.FloatTensor)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples
