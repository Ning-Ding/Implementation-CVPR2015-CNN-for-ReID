# -*- coding: utf-8 -*-
# --------------------------------------------------------
# Implementation-CVPR2015-CNN-for-ReID
# Copyright (c) 2017 Ning Ding
# Licensed under The MIT License [see LICENSE for details]
# Written by Ning Ding
# --------------------------------------------------------

"""
Data Preparation Script.
"""
import numpy as np
np.random.seed(1217)
import h5py
from PIL import Image
from keras.preprocessing import image as pre_image

#Reduce the batch size(ex: 50) if there is a resource exhaust error.
BATCH_SIZE = 150

class NumpyArrayIterator_for_CUHK03(pre_image.Iterator):
    
    def __init__(self, f, train_or_validation = 'train', flag = 1, image_data_generator = None,
                 batch_size = BATCH_SIZE, shuffle=True, seed=1217):
        self.f = f
        self.length = len(f['a'][train_or_validation].keys())
        self.train_or_validation = train_or_validation
        self.flag = flag
        self.image_data_generator = image_data_generator
        super(NumpyArrayIterator_for_CUHK03, self).__init__(3000000, batch_size / 2, shuffle, seed)

    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
            
        batch_x1 = np.zeros(tuple([current_batch_size * 2] + [160,60,3]))
        batch_x2 = np.zeros(tuple([current_batch_size * 2] + [160,60,3]))
        batch_y  = np.zeros([current_batch_size * 2, 2])
        
        for i, j in enumerate(index_array):
            
            k = np.random.randint(self.length)
            while k == 1155:
                k = np.random.randint(self.length)
            ja = np.random.randint(self.f['a'][self.train_or_validation][str(k)].shape[0])
            jb = np.random.randint(self.f['b'][self.train_or_validation][str(k)].shape[0])
            
            x1 = self.f['a'][self.train_or_validation][str(k)][ja]
            x2 = self.f['b'][self.train_or_validation][str(k)][jb]
            if np.random.rand() > self.flag:
                x1 = self.image_data_generator.random_transform(x1.astype('float32'))
            if np.random.rand() > self.flag:
                x2 = self.image_data_generator.random_transform(x2.astype('float32'))
            
            batch_x1[2*i] = x1
            batch_x2[2*i] = x2
            batch_y[2*i][1] = 1
            
            ka,kb = np.random.choice(range(self.length),2)
            while ka == 1155 or kb == 1155:
                ka,kb = np.random.choice(range(self.length),2)
                   
            ja = np.random.randint(self.f['a'][self.train_or_validation][str(ka)].shape[0])
            jb = np.random.randint(self.f['b'][self.train_or_validation][str(kb)].shape[0])
            
            x1 = self.f['a'][self.train_or_validation][str(ka)][ja]
            x2 = self.f['b'][self.train_or_validation][str(kb)][jb]
            
            batch_x1[2*i+1] = x1
            batch_x2[2*i+1] = x2
            batch_y[2*i+1][0] = 1
            
        return [batch_x1,batch_x2], batch_y


class ImageDataGenerator_for_multiinput(pre_image.ImageDataGenerator):
            
    def flow(self, f, train_or_validation = 'train', flag = 0, batch_size = BATCH_SIZE, shuffle=True, seed=1217):
        
        return NumpyArrayIterator_for_CUHK03(f, train_or_validation, flag, self, batch_size=batch_size, shuffle=shuffle, seed=seed)

    
    def agumentation(self, X, rounds=1, seed=None):
        
        if seed is not None:
            np.random.seed(seed)

        X = np.copy(X)
        aX = np.zeros(tuple([rounds * X.shape[0]] + list(X.shape)[1:]))
        for r in range(rounds):
            for i in range(X.shape[0]):
                aX[i + r * X.shape[0]] = self.random_transform(X[i])
        X = aX
        return X
