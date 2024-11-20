import numpy as np
import torch.nn as nn
import torch


class DatasetNumpy(nn.Module):
    def __init__(self, file_path, dataset_name, transform=None):
        super(DatasetNumpy, self).__init__()
        self.dataset = np.load(file_path, allow_pickle=True)
        self.transform = transform
        self.dataset_name = dataset_name
        targets = []
        for item in self.dataset:
            targets.append(item[1])
        self.targets = targets

    def __getitem__(self, index):
        # print(self.dataset[index])
        # input()
        x = self.dataset[index][0]
        y = self.dataset[index][1]

        if self.transform:
            x = self.transform(x)

        return x, y

    def __len__(self):
        return len(self.dataset)


    def __str__(self):
        return self.dataset_name


class DatasetNumpy2(nn.Module):
    def __init__(self, dataset_list, transform=None):
        super(DatasetNumpy2, self).__init__()
        self.dataset = dataset_list
        targets = []
        for item in self.dataset:
            targets.append(item[1])
        self.targets = targets
        self.transform = transform

    def __getitem__(self, index):
        # print(self.dataset[index])
        # input()
        x = self.dataset[index][0]
        y = self.dataset[index][1]

        if self.transform:
            x = self.transform(x)

        return x, y

    def __len__(self):
        return len(self.dataset)