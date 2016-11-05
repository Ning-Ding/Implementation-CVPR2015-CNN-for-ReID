# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 23:58:37 2016

@author: dingning
"""

import h5py
import numpy as np
from PIL import Image
from keras.preprocessing import image
'''
np.array(f[f[f['labeled'][0][i]][j][k]]).transpose(2,1,0)
this expression will get a numpy array of a picture with axis order is 'tf'
f[f['labeled'][0][i]][0].size
this expression return the numbers of id captured by i pari of cameras
i: from 0-4, the number i pair of cameras
j: from 0-9, the number j pictures of identity k captured by i pair of cameras
k: identity numbers
'''


def create_hdf5_dataset_for_cuhk03(file_path = './cuhk-03.mat'):
    with h5py.File(file_path) as fread, h5py.File('cuhk-03_for_CNN.h5') as fwrite:
        ftrain = fwrite.create_group('train')
        fvalid = fwrite.create_group('validation')
        ftests = fwrite.create_group('test')       
        val_index = (fread[fread['testsets'][0][0]][:].T - 1).tolist()
        tes_index = (fread[fread['testsets'][0][1]][:].T - 1).tolist()
        
        
        negative_index_list = []
        count_index = 0
        for ia in xrange(3):
            for ka in xrange(fread[fread['labeled'][0][ia]][0].size):        
                if [ia,ka] in val_index or [ia,ka] in tes_index:
                    continue
                else:
                    for ib in xrange(3):                        
                        for kb in xrange(fread[fread['labeled'][0][ib]][0].size):
                            if (ka == kb and ia == ib) or [ib,kb] in val_index or [ib,kb] in tes_index:
                                continue
                            else:
                                for ja in xrange(5):
                                    if len(fread[fread[fread['labeled'][0][ia]][ja][ka]].shape) == 3:
                                        for jb in xrange(5,10):
                                            if len(fread[fread[fread['labeled'][0][ib]][jb][kb]].shape) == 3:
                                                negative_index_list.append([[ia,ja,ka],[ib,jb,kb]])
                                                count_index += 1
                                                print ia,ja,ka,ib,jb,kb,'count:',count_index                                                
                                                break
                                        break
        print 'already found',len(negative_index_list),'negative pairs.'
        permutation_index = np.random.permutation(len(negative_index_list))  
        
        x1_array_list = []
        x2_array_list = []
        count = 0
        print 'loading positive data......'
        for i in xrange(3):
            for k in xrange(fread[fread['labeled'][0][i]][0].size):
                if [i,k] in val_index or [i,k] in tes_index:                    
                    continue            
                else:
                    for ja in xrange(5):
                        a = fread[fread[fread['labeled'][0][i]][ja][k]][:]
                        if a.size < 3:
                            continue
                        else:
                            for jb in xrange(5,10):
                                b = fread[fread[fread['labeled'][0][i]][jb][k]][:]
                                if b.size < 3:
                                    continue
                                else:
                                    x1_array_list.append(_resize_image(a))
                                    x2_array_list.append(_resize_image(b))
                                    count += 1
                                    print 'already load',count,'positive pairs'
                                    
        print 'already loaded all positive data.'
        print 'positive data number:',count
        pos_num = count * 5
        print 'positive data number after augmentation will be:',pos_num
        print 'total train number will be:',pos_num * 3
        print 'data augmentation begin......'
        Data_Generator = ImageDataGenerator_for_multiinput(width_shift_range=0.05,height_shift_range=0.05)
        print 'for the first input:'
        x_pos_1 = Data_Generator.fit(np.array(x1_array_list),augment=True,rounds=5,seed=1217)
        print 'for the second input:'
        x_pos_2 = Data_Generator.fit(np.array(x2_array_list),augment=True,rounds=5,seed=1217)
        print 'positive data augmentation done.'
        print 'positive data number after augmentation:',len(x_pos_1)
        print 'begin to store the data into local disk......'
        all_data_shuffle_index = np.random.permutation(pos_num * 3)
        x1_set = ftrain.create_dataset('x1',shape=(pos_num * 3,160,60,3))
        for i in xrange(len(x_pos_1)):
            x1_set[all_data_shuffle_index[i]] = x_pos_1[i]
            print 'already stored',i,'images in x1.'            
        del x_pos_1
        x2_set = ftrain.create_dataset('x2',shape=(pos_num * 3,160,60,3))
        for i in xrange(len(x_pos_2)):
            x2_set[all_data_shuffle_index[i]] = x_pos_2[i]
            print 'already stored',i,'images in x2.'        
        del x_pos_2
        y_set = ftrain.create_dataset('y',shape = (pos_num * 3,2))
        for i in xrange(pos_num):    
            y_set[all_data_shuffle_index[i]] = np.array([0,1])
            print 'already stored',i,'images in y.'
        print 'positive data already stored into local disk.'
                
        count = 0
        index_begin = pos_num
        for n in xrange(pos_num * 2):
            indexs = negative_index_list[permutation_index[n]]
            x1_set[all_data_shuffle_index[index_begin + count]] = _resize_image(fread[fread[fread['labeled'][0][indexs[0][0]]][indexs[0][1]][indexs[0][2]]][:])
            x2_set[all_data_shuffle_index[index_begin + count]] = _resize_image(fread[fread[fread['labeled'][0][indexs[1][0]]][indexs[1][1]][indexs[1][2]]][:])
            y_set[all_data_shuffle_index[index_begin + count]] = np.array([1,0])
            count += 1
            print 'already stored',count,'negative pairs'
            
        _make_validation_set(fread,fvalid)
        _make_test_set(fread,ftests)
        print 'Congratulations!!! All data already stored in local disk.'
        
        

def _resize_image(im_array,shape=(60,160)):
    if im_array.shape[2] > 3:
        im_array = im_array.transpose(2,1,0)
    im = Image.fromarray(im_array)
    im = im.resize(shape)
    array = np.array(im)
    return array/ 255.

def _make_validation_set(fread,fvalid):
    print 'Begin to make validation data set hdf5 file......' 
    val_index = fread[fread['testsets'][0][0]][:].T - 1
    x1_val_list = []
    x2_val_list = []
    y_val_list = []
    count_val_pos = 0
    print 'Positive Validation data making......'
    for i,k in val_index:
        for ja in xrange(5):
            if len(fread[fread[fread['labeled'][0][i]][ja][k]].shape) < 3:
                continue
            else:
                for jb in xrange(5,10):
                    if len(fread[fread[fread['labeled'][0][i]][jb][k]].shape) < 3:
                        continue
                    else:
                        x1_val_list.append(_resize_image(fread[fread[fread['labeled'][0][i]][ja][k]][:]))
                        x2_val_list.append(_resize_image(fread[fread[fread['labeled'][0][i]][jb][k]][:]))
                        y_val_list.append([0,1])
                        count_val_pos += 1
                        print 'already load',count_val_pos,'validation positive pairs'
    
    print 'positive validation pairs loaded,totally:',count_val_pos,'pairs, Begin to load negative validation pairs.'
    count_val_neg = 0
        
    negative_index_list = []
    for ia,ka in val_index:
        for ib,kb in val_index:
            if ia == ib and ka == kb:
                continue
            else:
                for ja in xrange(5):
                    if len(fread[fread[fread['labeled'][0][ia]][ja][ka]].shape) == 3:
                        for jb in xrange(5,10):
                            if len(fread[fread[fread['labeled'][0][ib]][jb][kb]].shape) == 3:
                                negative_index_list.append([[ia,ja,ka],[ib,jb,kb]])
                                break
                        break
                        
    print 'already found',len(negative_index_list),'negative pairs for validation set.'
    permutation_index = np.random.permutation(len(negative_index_list)) 
    
    for n in xrange(count_val_pos * 2):
        indexs = negative_index_list[permutation_index[n]]
        x1_val_list.append(_resize_image(fread[fread[fread['labeled'][0][indexs[0][0]]][indexs[0][1]][indexs[0][2]]][:]))
        x2_val_list.append(_resize_image(fread[fread[fread['labeled'][0][indexs[1][0]]][indexs[1][1]][indexs[1][2]]][:]))
        y_val_list.append([1,0])
        count_val_neg += 1
        print 'already load',count_val_neg,'validation negative pairs'
    
    val_data_shuffle_index = np.random.permutation(count_val_pos * 3)
    x1_set = fvalid.create_dataset('x1',shape=(count_val_pos * 3,160,60,3)) 
    x2_set = fvalid.create_dataset('x2',shape=(count_val_pos * 3,160,60,3))
    y_set = fvalid.create_dataset('y',shape=(count_val_pos * 3,2))
    for i in xrange(count_val_pos * 3):
        x1_set[val_data_shuffle_index[i]] = x1_val_list[i]
        x2_set[val_data_shuffle_index[i]] = x2_val_list[i]
        y_set[val_data_shuffle_index[i]] = y_val_list[i]
        print 'already stored',i,'validation image pairs.'
    print 'validation data already stored into local disk.'
    return 
                                        

def _make_test_set(fread,ftests):
    tes_index = fread[fread['testsets'][0][1]][:].T - 1
    list_a = []
    list_b = []
    for i,k in tes_index:
        for ja in xrange(5):
            a = fread[fread[fread['labeled'][0][i]][ja][k]][:]
            if a.size < 3:
                continue
            else:
                for jb in xrange(5,10):                    
                    b = fread[fread[fread['labeled'][0][i]][jb][k]][:]
                    if b.size < 3:
                        continue
                    else:
                        list_a.append(_resize_image(a))
                        list_b.append(_resize_image(b))
                        print 'already loaded',len(list_a),'test pairs.'
                        break
                break
    ftests.create_dataset('a',data = np.array(list_a))
    ftests.create_dataset('b',data = np.array(list_b))
    print 'test data already in local disk.'
    return




class ImageDataGenerator_for_multiinput(image.ImageDataGenerator):
    def fit(self, X,
            augment=False,
            rounds=1,
            seed=None):
        '''
        # Arguments
            X: Numpy array, the data to fit on.
            augment: whether to fit on randomly augmented samples
            rounds: if `augment`,
                how many augmentation passes to do over the data
            seed: random seed.
        '''
        if seed is not None:
            np.random.seed(seed)

        X = np.copy(X)
        if augment:
            aX = np.zeros(tuple([rounds * X.shape[0]] + list(X.shape)[1:]))
            print 'total rounds number:',rounds
            for r in range(rounds):
                print 'rounds:',r+1
                for i in range(X.shape[0]):
                    aX[i + r * X.shape[0]] = self.random_transform(X[i])
            X = aX

        return X


if __name__ == '__main__':
    create_hdf5_dataset_for_cuhk03()






                
