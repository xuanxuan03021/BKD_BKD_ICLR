import sys
# sys.path.append("/home/mengxuan/BackdoorBox")
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import numpy as np
import core
import tensorflow as tf


# config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
# print(config)
# sess = tf.compat.v1.Session(config=config)

import random
import matplotlib.pyplot as plt
import cv2


import albumentations
from scipy.fftpack import dct, idct

import math
import torch
from torch.nn import functional as F
import torchvision.models as models
# from core.models.model_all import ResNet as ResNet
# from core.models.model_all import DetectorCNN as DetectorCNN

# rewrite it to pytorch model
class detector_model(nn.Module):
    def __init__(self, structure=18, class_num=100, input_size=32,label_number=10):
        super(detector_model, self).__init__()
        if input_size == 32 :
            self.normal_model = ResNet(structure, class_num)
        elif input_size == 64:
            self.normal_model = models.resnet18(pretrained=False)
            self.normal_model.fc = torch.nn.Linear(512, class_num)
        if input_size == 224:
            print("======use efficientnet!!!!")
            from efficientnet_pytorch import EfficientNet
            # self.normal_model = EfficientNet.from_name('efficientnet-b0', num_classes=class_num)
            self.normal_model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=class_num)

        if input_size == 32:
            self.attack_model = DetectorCNN(num_classes=class_num, input_size=input_size)
        elif input_size == 224:
            self.attack_model = DetectorCNN_ImageNet(num_classes=class_num, input_size=input_size)
        self.concat_layer =  nn.Linear(2*class_num, label_number)

    def forward(self, x,dct_x):

        # print(x)
        # input()
        #use normal resnet model to get the normal embedding
        normal_emb = self.normal_model(x)
        # print(normal_emb)
        #
        # print(x.shape)
        # print(dct_x.shape)
        # print("-----")
        # input()

        #use dct version of x to get the attack embedding
        attack_emb = self.attack_model(dct_x)
        # print(attack_emb)
        # input()

        #concatenate the two embeddings
        concat_emb = torch.cat((normal_emb, attack_emb), dim=1)
        # print(concat_emb.shape)

        #get the final output
        output = self.concat_layer(concat_emb)

        return output



class detector_model_final_layer(nn.Module):
    def __init__(self, structure=18,input_size=32,label_number=10):
        super(detector_model_final_layer, self).__init__()
        if input_size == 32 :
            self.normal_model = ResNet(structure, label_number)
        elif input_size == 64 or input_size == 224:
            self.normal_model = models.resnet18(pretrained=False)
            self.normal_model.fc = torch.nn.Linear(512, label_number)
        if input_size == 224:
            from efficientnet_pytorch import EfficientNet
            self.normal_model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=label_number)

        self.attack_model = DetectorCNN(num_classes=label_number, input_size=input_size)
        self.concat=nn.Softmax(dim=1)

    def forward(self, x,dct_x,alpha=0.5):


        #use normal resnet model to get the normal embedding
        normal_emb = self.normal_model(x)

        #use dct version of x to get the attack embedding
        attack_emb = self.attack_model(dct_x)

        # print(attack_emb.shape)
        # print(normal_emb.shape)

        #softmax the two embeddings
        output = self.concat(alpha* normal_emb+ (1-alpha)*attack_emb)

        return output

def train_detector(x_final_train,y_final_train,dataset_name, structure=18, class_num=100, input_size=32,label_number=10, num_epochs=5, batch_size=64, learning_rate=0.05, weight_decay = 1e-4  # Note: This is used in Keras as L2 regularization, handled differently in PyTorch
):
    # Model initialization using gpu if available

    model = detector_model(structure, class_num, input_size,label_number)
    model.train()
    model = model.cuda()

    #model parallel
    model = nn.DataParallel(model, device_ids=device_ids)

    optimizer = optim.Adadelta(model.module.parameters(), lr=learning_rate,weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    # Check if model checkpoint exists
    model_path = './detector/8_CNN_{}.pth'.format(dataset_name)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        return model

    # Data preparation
    for epoch in tqdm(range(num_epochs)):        # Convert to PyTorch tensors
        train_data = torch.tensor(x_final_train, dtype=torch.float32)
        train_labels = torch.tensor(y_final_train, dtype=torch.long)

        # Create DataLoader
        train_dataset = TensorDataset(train_data, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Training loop
        model.train()
        for i, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data.cuda())
            loss = criterion(output, target.cuda())
            loss.backward()
            optimizer.step()

    # Save the trained model
    torch.save(model.state_dict(), model_path)

    return model


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class _ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(_ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet(num, num_classes=10):
    if num == 18:
        return _ResNet(BasicBlock, [2,2,2,2], num_classes)
    elif num == 34:
        return _ResNet(BasicBlock, [3,4,6,3], num_classes)
    elif num == 50:
        return _ResNet(Bottleneck, [3,4,6,3], num_classes)
    elif num == 101:
        return _ResNet(Bottleneck, [3,4,23,3], num_classes)
    elif num == 152:
        return _ResNet(Bottleneck, [3,8,36,3], num_classes)
    else:
        raise NotImplementedError



class DetectorCNN(nn.Module):
    def __init__(self, num_classes=100, input_size=32):
        super(DetectorCNN, self).__init__()

        # Define the layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding='same', bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding='same', bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding='same', bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding='same', bias=False)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(0.3)

        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding='same', bias=False)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding='same', bias=False)
        self.bn6 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout(0.4)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128 * (input_size // 8) * (input_size // 8),
                            num_classes)  # Adjust input size based on your data

    def forward(self, x):
        x = self.bn1(F.elu(self.conv1(x)))
        x = self.bn2(F.elu(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout1(x)

        x = self.bn3(F.elu(self.conv3(x)))
        x = self.bn4(F.elu(self.conv4(x)))
        x = self.pool2(x)
        x = self.dropout2(x)

        x = self.bn5(F.elu(self.conv5(x)))
        x = self.bn6(F.elu(self.conv6(x)))
        x = self.pool3(x)
        x = self.dropout3(x)

        x = self.flatten(x)
        x = self.fc(x)
        return x

class DetectorCNN_ImageNet(nn.Module):
    def __init__(self, num_classes=100, input_size=32):
        super(DetectorCNN_ImageNet, self).__init__()

        # Define the layers
        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2)
        self.bn1 = nn.BatchNorm2d(96)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(256)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.dropout1 = nn.Dropout(0.2)

        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(384)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(384)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.dropout2 = nn.Dropout(0.3)

        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.dropout3 = nn.Dropout(0.4)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(9216,
                            num_classes)  # Adjust input size based on your data

    def forward(self, x):
        x = self.bn1(F.elu(self.conv1(x)))
        x = self.bn2(F.elu(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout1(x)

        x = self.bn3(F.elu(self.conv3(x)))
        x = self.bn4(F.elu(self.conv4(x)))
        x = self.pool2(x)
        x = self.dropout2(x)

        x = self.bn5(F.elu(self.conv5(x)))
        x = self.bn6(F.elu(self.conv6(x)))
        x = self.pool3(x)
        x = self.dropout3(x)

        x = self.flatten(x)
        # print(x.shape)
        x = self.fc(x)
        return x