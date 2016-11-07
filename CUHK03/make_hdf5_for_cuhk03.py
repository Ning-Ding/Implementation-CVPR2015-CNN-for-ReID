# -*- coding: utf-8 -*-
"""
Created on Mon Nov 07 13:31:35 2016

@author: dingning
"""

import h5py
import numpy as np

'''
np.array(f[f[f['labeled'][0][i]][j][k]]).transpose(2,1,0)
this expression will get a numpy array of a picture with axis order is 'tf'
f[f['labeled'][0][i]][0].size
this expression return the numbers of id captured by i pari of cameras
i: from 0-4, the number i pair of cameras
j: from 0-9, the number j pictures of identity k captured by i pair of cameras
k: identity numbers
'''

def make_hdf5_for_cuhk03(file_path = '/home/lpc/dataset/cuhk-03.mat'):
    with h5py.File(file_path,'r') as f:
        val_index = (f[f['testsets'][0][0]][:].T - 1).tolist()
        tes_index = (f[f['testsets'][0][1]][:].T - 1).tolist()
        
        index_train = []
        index_valid = []
        index_tests = []
        for i in xrange(3):
            for k in xrange(f[f['labeled'][0][i]][0].size):
                if [i,k] in val_index:
                    index_valid.append([i,k])
                elif [i,k] in tes_index:
                    index_tests.append([i,k])
                else:
                    index_train.append([i,k])
                    
        index_train = np.array(index_train)
        index_valid = np.array(index_valid)
        index_tests = np.array(index_tests)
        np.random.shuffle(index_train)
        np.random.shuffle(index_valid)
        np.random.shuffle(index_tests)            
        with h5py.File('cuhk-03_for_train.h5') as fw:
            fn = fw.create_group('negative')
            fn.create_dataset('train',data = index_train)
            fn.create_dataset('validation',data = index_valid)
            fn.create_dataset('test',data = index_tests)
        
        
        index_list = []
        for i in xrange(3):
            for k in xrange(f[f['labeled'][0][i]][0].size):
                if [i,k] in val_index or [i,k] in tes_index:
                    continue
                for ja in xrange(5):
                    if len(f[f[f['labeled'][0][i]][ja][k]].shape) == 3:
                        for jb in xrange(5,10):
                            if len(f[f[f['labeled'][0][i]][jb][k]].shape) == 3:
                                index_list.append([i,k,ja,jb])
                                print len(index_list)
        
        index_list = np.array(index_list)
        np.random.shuffle(index_list)                    
        with h5py.File('cuhk-03_for_train.h5') as fw:
            fp = fw.create_group('positive')
            fp.create_dataset('train',data = index_list)
        
        index_list = []
        for i,k in val_index:
            for ja in xrange(5):
                if len(f[f[f['labeled'][0][i]][ja][k]].shape) == 3:
                    for jb in xrange(5,10):
                        if len(f[f[f['labeled'][0][i]][jb][k]].shape) == 3:
                            index_list.append([i,k,ja,jb])
                            print len(index_list)
                            
        index_list = np.array(index_list)
        np.random.shuffle(index_list)                    
        with h5py.File('cuhk-03_for_train.h5') as fw:
            fw['positive'].create_dataset('validation',data = index_list)
            
        index_list = []
        for i,k in tes_index:
            for ja in xrange(5):
                if len(f[f[f['labeled'][0][i]][ja][k]].shape) == 3:
                    for jb in xrange(5,10):
                        if len(f[f[f['labeled'][0][i]][jb][k]].shape) == 3:
                            index_list.append([i,k,ja,jb])
                            print len(index_list)
                            
        index_list = np.array(index_list)
        np.random.shuffle(index_list)                    
        with h5py.File('cuhk-03_for_train.h5') as fw:
            fw['positive'].create_dataset('test',data = index_list)
        
if __name__ == '__main__':
    user_name = raw_input('please input your system user name:')
    if user_name == 'lpc':
        make_hdf5_for_cuhk03()
    else:
        make_hdf5_for_cuhk03('/home/'+user_name+'/dataset/cuhk-03.mat')
                    
