# import keras
import torch
# from keras.models import Sequential
# from keras.utils import np_utils
# from keras.preprocessing.image import ImageDataGenerator
# from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
# from keras.layers import Conv2D, MaxPooling2D
# from keras.datasets import cifar10
# from keras import regularizers
# from keras.callbacks import LearningRateScheduler
import numpy as np
from tqdm import tqdm

from .base import Base
import cv2

class STRIP(Base):
    def __init__(self, x_train, model, poisoned_testset, clean_testset, device, seed=0,
                 deterministic=False):
        super(STRIP, self).__init__(seed, deterministic)
        self.x_train = x_train
        self.model = model
        self.poisoned_testset = poisoned_testset
        self.clean_testset = clean_testset
        self.device = device

    def superimpose(self, background, overlay):
        
        added_image = cv2.addWeighted(background, 1, overlay, 1, 0)
        return (added_image.reshape(32, 32, 3))

    def entropyCal(self, background, n):
        entropy_sum = [0] * n
        x1_add = [0] * n
        index_overlay = np.random.randint(0, len(self.x_train), size=n)
        for x in range(n):
            x1_add[x] = (self.superimpose(background, self.x_train[index_overlay[x]]))
            # import matplotlib.pyplot as plt
            # plt.imshow(self.superimpose(background, self.x_train[index_overlay[x]]))
            # plt.show()
            #
            # plt.imshow(self.x_train[index_overlay[x]])
            # plt.show()
            # input()

        py1_add = self.model(torch.Tensor(x1_add).permute(0, 3, 1, 2).to(self.device)).detach().cpu().numpy()
        EntropySum = -np.nansum(py1_add * np.log2(py1_add))
        return EntropySum


    def cal_score(self):
        n_test = len(self.clean_testset)
        n_sample = 6
        entropy_benigh = [0] * n_test
        entropy_trojan = [0] * n_test

        for j in tqdm(range(n_test)):
            x_background = self.clean_testset[j][0].cpu().numpy().transpose(1,2,0)
            entropy_benigh[j] = self.entropyCal(x_background, n_sample)

        for j in tqdm(range(n_test)):
            x_poison = self.poisoned_testset[j][0].cpu().numpy().transpose(1,2,0)
            entropy_trojan[j] = self.entropyCal(x_poison, n_sample)

        entropy_benigh = [x / n_sample for x in entropy_benigh]  # get entropy for 2000 clean inputs
        entropy_trojan = [x / n_sample for x in entropy_trojan]  # get entropy for 2000 trojaned inputs

        return entropy_trojan, entropy_benigh
