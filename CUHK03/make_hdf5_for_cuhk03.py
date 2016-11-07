# -*- coding: utf-8 -*-
"""
Created on Mon Nov 07 17:04:05 2016

@author: dingning
"""
import h5py
import numpy as np
from PIL import Image

def make_hdf5_for_cuhk03(file_path = '/home/lpc/dataset/cuhk-03.mat'):
    
    with h5py.File(file_path,'r') as f, h5py.File('cuhk-03.h5') as fw:
        val_index = (f[f['testsets'][0][0]][:].T - 1).tolist()
        tes_index = (f[f['testsets'][0][1]][:].T - 1).tolist()
        
        fwa = fw.create_group('a')
        fwb = fw.create_group('b')
        fwat = fwa.create_group('train')
        fwav = fwa.create_group('validation')
        fwae = fwa.create_group('test')
        fwbt = fwb.create_group('train')
        fwbv = fwb.create_group('validation')
        fwbe = fwb.create_group('test')
        
        temp = []
        count_t = 0
        count_v = 0
        count_e = 0
        for i in xrange(3):
            for k in xrange(f[f['labeled'][0][i]][0].size):
                print i,k
                if [i,k] in val_index:
                    for j in xrange(5):
                        if len(f[f[f['labeled'][0][i]][j][k]].shape) == 3:
                            temp.append(np.array((Image.fromarray(f[f[f['labeled'][0][i]][j][k]][:].transpose(2,1,0))).resize((64,128))) / 255.)
                    fwav.create_dataset(str(count_v),data = np.array(temp))
                    temp = []
                    for j in xrange(5,10):
                        if len(f[f[f['labeled'][0][i]][j][k]].shape) == 3:
                            temp.append(np.array((Image.fromarray(f[f[f['labeled'][0][i]][j][k]][:].transpose(2,1,0))).resize((64,128))) / 255.)
                    fwbv.create_dataset(str(count_v),data = np.array(temp))
                    temp = []
                    count_v += 1
                elif [i,k] in tes_index:
                    for j in xrange(5):
                        if len(f[f[f['labeled'][0][i]][j][k]].shape) == 3:
                            temp.append(np.array((Image.fromarray(f[f[f['labeled'][0][i]][j][k]][:].transpose(2,1,0))).resize((64,128))) / 255.)
                    fwae.create_dataset(str(count_e),data = np.array(temp))
                    temp = []
                    for j in xrange(5,10):
                        if len(f[f[f['labeled'][0][i]][j][k]].shape) == 3:
                            temp.append(np.array((Image.fromarray(f[f[f['labeled'][0][i]][j][k]][:].transpose(2,1,0))).resize((64,128))) / 255.)
                    fwbe.create_dataset(str(count_e),data = np.array(temp))
                    temp = []
                    count_e += 1
                else:
                    for j in xrange(5):
                        if len(f[f[f['labeled'][0][i]][j][k]].shape) == 3:
                            temp.append(np.array((Image.fromarray(f[f[f['labeled'][0][i]][j][k]][:].transpose(2,1,0))).resize((64,128))) / 255.)
                    fwat.create_dataset(str(count_t),data = np.array(temp))
                    temp = []
                    for j in xrange(5,10):
                        if len(f[f[f['labeled'][0][i]][j][k]].shape) == 3:
                            temp.append(np.array((Image.fromarray(f[f[f['labeled'][0][i]][j][k]][:].transpose(2,1,0))).resize((64,128))) / 255.)
                    fwbt.create_dataset(str(count_t),data = np.array(temp))
                    temp = []
                    count_t += 1
                    
if __name__ == '__main__':
    user_name = raw_input('please input your system user name:')
    if user_name == 'lpc':
        make_hdf5_for_cuhk03()
    else:
        make_hdf5_for_cuhk03('/home/'+user_name+'/dataset/cuhk-03.mat')                    
