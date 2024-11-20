import LAVA.lava as lava

import torch
import torchvision
# print(torch.__version__)
# print(torchvision.__version__)
# print(torch.version.cuda)
# print(torch.cuda.is_available())

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from torch.autograd import Variable

import matplotlib.pyplot as plt
from torch import tensor
from torchvision import datasets, transforms
import pandas as pd
import numpy as np
from copy import deepcopy as dpcp

from torch.utils.data import Dataset, TensorDataset, DataLoader

cuda_num = 0
import torchvision
print(torchvision.__version__)
import torch
print(torch.__version__)
import os
os.environ["CUDA_VISIBLE_DEVICES"]=str(cuda_num)
print(os.environ["CUDA_VISIBLE_DEVICES"])
torch.cuda.set_device(cuda_num)
print("Cuda device: ", torch.cuda.current_device())
print("cude devices: ", torch.cuda.device_count())
device = torch.device('cuda:' + str(cuda_num) if torch.cuda.is_available() else 'cpu')

training_size = 40000
valid_size = 10000
resize = 32
portion = 0.25

loaders, shuffle_ind = lava.load_data_corrupted(corrupt_type='shuffle', dataname='CIFAR10', resize=resize,
                                        training_size=training_size, test_size=valid_size, currupt_por=portion)


feature_extractor = lava.load_pretrained_feature_extractor('cifar10_embedder_preact_resnet18.pth', device)


dual_sol, trained_with_flag = lava.compute_dual(feature_extractor, loaders['train'], loaders['test'],
                                                training_size, shuffle_ind, resize=resize)