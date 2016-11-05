# -*- coding: utf-8 -*-
"""
Created on Sat Nov 05 21:53:32 2016

@author: dingning
"""
import os
import h5py
import numpy as np


def make_hdf5_index_for_market1501(train_or_test = 'train'):
    f = h5py.File('market1501_index.h5')
    path_list = get_image_path_list(train_or_test = train_or_test)
    total = len(path_list) ** 2
    pos_index = []
    neg_index = []
    count = 0
    for path1 in path_list:
        for path2 in path_list:
            count += 1
            print 'already processed',count,'paths in',total,'paths.'
            if compare_path == 2:
                neg_index.append([path1,path2])
            elif compare_path == 1:
                pos_index.append([path1,path2])
    print 'totally',len(pos_index),'positive pairs.'
    print 'totally',len(neg_index),'negative pairs.'    
    pos_index = np.array(pos_index)
    np.random.shuffle(pos_index)
    neg_index = np.array(neg_index)
    np.random.shuffle(neg_index)
    f_train = f.create_group(train_or_test)
    f_train.create_dataset('positive',data=pos_index)
    f_train.create_dataset('negative',data=neg_index)
    

def compare_path(path1,path2):
    if path1[6] == path2[6]:
        return 0
    else:
        if path1[0:4] == path2[0:4]:
            return 1
        else:
            return 2


def get_image_path_list(train_or_test = 'train'):
    '''
    get a list containing all the paths of images in the trainset
    ---------------------------------------------------------------------------
    INPUT:
        parameters: model parameter object
    OUTPUT:
        a list with all the images' paths
    ---------------------------------------------------------------------------
    '''
    if train_or_test == 'train':
        folder_path = '~/dataset/market1501/boundingboxtrain'
    else:
        folder_path = '~/dataset/market1501/boundingboxtest'
    assert os.path.exists(folder_path)
    assert os.path.isdir(folder_path)
    print 'already get all the image path.'
    return os.listdir(folder_path)
    
    
if __name__ == '__main__':
    make_hdf5_index_for_market1501('train')
    make_hdf5_index_for_market1501('test')