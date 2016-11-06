# -*- coding: utf-8 -*-
"""
Created on Sat Nov 05 21:53:32 2016

@author: dingning
"""
import os
import h5py
import numpy as np

def make_positive_index_market1501(train_or_test = 'train',user_name = 'ubuntu'):
    f = h5py.File('market1501_positive_index.h5')
    path_list = get_image_path_list(train_or_test = train_or_test, system_user_name = user_name)
    index = []
    i = 0
    while i < len(path_list):
        j = i + 1
        while j < len(path_list) and path_list[j][6] == path_list[i][6]:
            j += 1
        i = j - 1
        while j < len(path_list) and path_list[j][0:4] == path_list[i][0:4]:
            if path_list[j][6] != path_list[i][6]:
                index.append([path_list[i],path_list[j]])
                index.append([path_list[j],path_list[i]])
                print len(index)
            j += 1
        i += 1
    print 'transforming the list to the numpy array......'
    index = np.array(index)
    print 'shuffling the numpy array......'
    np.random.shuffle(index)
    print 'storing the array to HDF5 file......'
    f.create_dataset(train_or_test,data = index)


def get_image_path_list(train_or_test = 'train',system_user_name = 'ubuntu'):
    if train_or_test == 'train':
        folder_path = '/home/' + system_user_name + '/dataset/market1501/boundingboxtrain'
    else:
        folder_path = '/home/' + system_user_name + '/dataset/market1501/boundingboxtest'
    assert os.path.isdir(folder_path)
    print 'already get all the image path.'
    return sorted(os.listdir(folder_path))
    
    
if __name__ == '__main__':
    user_name = raw_input('input your system user name:')
    make_positive_index_market1501('train',user_name=user_name)
    make_positive_index_market1501('test',user_name=user_name)
