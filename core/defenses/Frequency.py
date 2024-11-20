import os

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers
from keras.optimizers import experimental
import numpy as np

import tensorflow as tf

from .base import Base

# config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
# sess = tf.compat.v1.Session(config=config)

import random
import matplotlib.pyplot as plt
import cv2


import albumentations
from scipy.fftpack import dct, idct

import math
import torch
from torch.nn import functional as F





class Frequency(Base):
    def __init__(self, args,
                 poisoned_testset, clean_testset, clean_trainset, seed=0,
                 deterministic=False):
        """

        Args:
            poisoned_testset: Dataset to be verified
            clean_trainset: a small batch of data that can be obtained for training frequency detector
            seed:
            deterministic:
        """
        super(Frequency, self).__init__(seed, deterministic)
        self.poisoned_testset = poisoned_testset
        self.trainset = clean_trainset
        self.clean_testset = clean_testset
        self.args = args

    def tensor2img(self, t):
        t_np = t.detach().cpu().numpy().transpose(1, 2, 0)
        return t_np

    def gauss_smooth(self, image, sig=6):
        size_denom = 5.
        sigma = sig * size_denom
        kernel_size = sigma
        mgrid = np.arange(kernel_size, dtype=np.float32)
        mean = (kernel_size - 1.) / 2.
        mgrid = mgrid - mean
        mgrid = mgrid * size_denom
        kernel = 1. / (sigma * math.sqrt(2. * math.pi)) * \
                 np.exp(-(((mgrid - 0.) / (sigma)) ** 2) * 0.5)
        kernel = kernel / np.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernelx = np.tile(np.reshape(kernel, (1, 1, int(kernel_size), 1)), (3, 1, 1, 1))
        kernely = np.tile(np.reshape(kernel, (1, 1, 1, int(kernel_size))), (3, 1, 1, 1))

        padd0 = int(kernel_size // 2)
        evenorodd = int(1 - kernel_size % 2)

        pad = torch.nn.ConstantPad2d((padd0 - evenorodd, padd0, padd0 - evenorodd, padd0), 0.)
        in_put = torch.from_numpy(np.expand_dims(np.transpose(image.astype(np.float32), (2, 0, 1)), axis=0))
        output = pad(in_put)

        weightx = torch.from_numpy(kernelx)
        weighty = torch.from_numpy(kernely)
        conv = F.conv2d
        output = conv(output, weightx, groups=3)
        output = conv(output, weighty, groups=3)
        output = self.tensor2img(output[0])

        return output

    def dct2(self, block):
        return dct(dct(block.T, norm='ortho').T, norm='ortho')

    def idct2(self, block):
        return idct(idct(block.T, norm='ortho').T, norm='ortho')

    def valnear0(self, dct_ori, rmin=-1.5, rmax=1.5):
        return len(dct_ori[dct_ori < rmax][dct_ori[dct_ori < rmax] > rmin])

    def addnoise(self, img):
        aug = albumentations.GaussNoise(p=1, mean=25, var_limit=(10, 70))
        augmented = aug(image=(img * 255).astype(np.uint8))
        auged = augmented['image'] / 255
        return auged

    def randshadow(self, img):
        aug = albumentations.RandomShadow(p=1)
        test = (img * 255).astype(np.uint8)
        augmented = aug(image=cv2.resize(test, (32, 32)))
        auged = augmented['image'] / 255
        return auged

    def normalization(self, data):
        _range = np.max(data) - np.min(data)
        return (data - np.min(data)) / _range


    def patching_train(self, clean_sample):
        '''
         this code conducts a patching procedure with random white blocks or random noise block
         '''
        attack = np.random.randint(0, 5)
        pat_size_x = np.random.randint(2, 8)
        pat_size_y = np.random.randint(2, 8)
        output = np.copy(clean_sample)
        if attack == 0:
            block = np.ones((pat_size_x, pat_size_y, 3))
        elif attack == 1:
            block = np.random.rand(pat_size_x, pat_size_y, 3)
        elif attack == 2:
            return self.addnoise(output)
        elif attack == 3:
            return self.randshadow(output)
        if attack == 4:
            randind = np.random.randint(self.trainset.shape[0])
            tri = self.trainset[randind]
            # print(tri.shape, output.shape)
            mid = output + 0.3 * tri
            mid[mid > 1] = 1
            return mid

        margin = np.random.randint(0, 6)
        rand_loc = np.random.randint(0, 4)
        if rand_loc == 0:
            output[margin:margin + pat_size_x, margin:margin + pat_size_y, :] = block  # upper left
        elif rand_loc == 1:
            output[margin:margin + pat_size_x, 32 - margin - pat_size_y:32 - margin, :] = block
        elif rand_loc == 2:
            output[32 - margin - pat_size_x:32 - margin, margin:margin + pat_size_y, :] = block
        elif rand_loc == 3:
            output[32 - margin - pat_size_x:32 - margin, 32 - margin - pat_size_y:32 - margin,
            :] = block  # right bottom

        output[output > 1] = 1
        return output

    def patching_test(self, clean_sample, attack_name):
        '''
        this code conducts a patching procedure to generate backdoor data
        **please make sure the input sample's label is different from the target label

        clean_sample: clean input
        attack_name: trigger's file name
        '''

        if attack_name == 'badnets':
            output = np.copy(clean_sample)
            pat_size = 4
            output[32 - 1 - pat_size:32 - 1, 32 - 1 - pat_size:32 - 1, :] = 1

        else:
            if attack_name == 'l0_inv':
                trimg = plt.imread('./triggers/' + attack_name + '.png')
                mask = 1 - np.transpose(np.load('./triggers/mask.npy'), (1, 2, 0))
                output = clean_sample * mask + trimg
            elif attack_name == 'smooth':
                trimg = np.load('./triggers/best_universal.npy')[0]
                output = clean_sample + trimg
                output = self.normalization(output)
            else:
                trimg = plt.imread('./triggers/' + attack_name + '.png')
                output = clean_sample + trimg
        output[output > 1] = 1
        return output

    def get_detector(self):
        # Simple 6-layer CNN
        weight_decay = 1e-4
        num_classes = 2
        model = Sequential()
        model.add(Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay),
                         input_shape=self.trainset.shape[1:]))
        model.add(Activation('elu'))
        model.add(BatchNormalization())
        model.add(Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('elu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('elu'))
        model.add(BatchNormalization())
        model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('elu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.3))

        model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('elu'))
        model.add(BatchNormalization())
        model.add(
            Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), name='last_conv'))
        model.add(Activation('elu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.4))

        model.add(Flatten())
        model.add(Dense(num_classes, activation='softmax', name='dense'))

        return model

    def train_detector(self):

        opt = experimental.Adadelta(lr=0.05)

        detector = self.get_detector()
        if os.path.exists('./detector/6_CNN_{}.h5py'.format(self.args.dataset)):
            detector.load_weights('./detector/6_CNN_{}.h5py'.format(self.args.dataset))
            return detector


        detector.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

        import tqdm
        for i in tqdm.tqdm(range(5)):
            poi_train = np.zeros_like(self.trainset)
            for i in range(self.trainset.shape[0]):
                poi_train[i] = self.patching_train(self.trainset[i])

            # 3channel dct
            x_dct_train = np.vstack((self.trainset, poi_train))
            y_dct_train = (np.vstack((np.zeros((self.trainset.shape[0], 1)), np.ones((self.trainset.shape[0], 1))))).astype(np.int8)
            for i in range(x_dct_train.shape[0]):
                for channel in range(3):
                    x_dct_train[i][:, :, channel] = self.dct2((x_dct_train[i][:, :, channel] * 255).astype(np.uint8))

            # SHUFFLE TRAINING DATA
            idx = np.arange(x_dct_train.shape[0])
            random.shuffle(idx)
            x_final_train = x_dct_train[idx]
            y_final_train = y_dct_train[idx]
            hot_lab = np.squeeze(np.eye(2)[y_final_train])

            detector.fit(x_final_train, hot_lab, epochs=5, batch_size=64)
            detector.save('./detector/6_CNN_{}.h5py'.format(self.args.dataset))

        return detector

    def verify(self):
        detector = self.train_detector()

        opt = experimental.Adadelta(lr=0.05)

        detector.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

        poi_test = np.zeros((len(self.poisoned_testset), 32, 32, 3))
        for idx, (img, target) in enumerate(self.poisoned_testset):
            poi_test[idx] = img.cpu().numpy().transpose(1,2,0)

        x_dct_test = poi_test  # [:,:,:,0]
        for i in range(x_dct_test.shape[0]):
            for channel in range(3):
                x_dct_test[i][:, :, channel] = self.dct2((x_dct_test[i][:, :, channel] * 255).astype(np.uint8))

        poi_result = detector.predict(x_dct_test)

        clean_test = np.zeros((len(self.clean_testset), 32, 32, 3))
        for idx, (img, target) in enumerate(self.clean_testset):
            clean_test[idx] = img.cpu().numpy().transpose(1,2,0)

        x_dct_test = clean_test  # [:,:,:,0]
        for i in range(x_dct_test.shape[0]):
            for channel in range(3):
                x_dct_test[i][:, :, channel] = self.dct2((x_dct_test[i][:, :, channel] * 255).astype(np.uint8))

        clean_result = detector.predict(x_dct_test)


        return list(poi_result[:, 1]), list(clean_result[:,1])

