# -*- coding: utf-8 -*-
"""
Created on Sat Nov 05 21:53:32 2016

@author: dingning
"""
import os
import h5py
import numpy as np
from PIL import Image

def make_positive_index_market1501(train_or_test = 'train',user_name = 'ubuntu'):
    f = h5py.File('market1501_positive_index.h5')
    path_list = get_image_path_list(train_or_test = train_or_test, system_user_name = user_name)
    index = []
    i = 0
    while i < len(path_list):
        j = i + 1
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


def make_test_hdf5(user_name='ubuntu'):
    with h5py.File('test.h5') as f:
        path_list = get_image_path_list(train_or_test='test')
        i = path_list[0][0:4]
        c = path_list[0][6]
        temp = []
        fi = f.create_group(i)
        for path in path_list:
            print path
            if path[0:4] == i:
                if path[6] == c:
                    temp.append(np.array(Image.open('/home/' + user_name + '/dataset/market1501/boundingboxtest/' + path)))
                else:
                    print len(temp)
                    fi.create_dataset(c,data = np.array(temp))
                    temp = []
                    c = path[6]
                    print c
                    temp.append(np.array(Image.open('/home/' + user_name + '/dataset/market1501/boundingboxtest/' + path)))
            else:
                fi.create_dataset(c,data = np.array(temp))
                i = path[0:4]
                fi = f.create_group(i)                
                print i
                c = path[6]
                print c
                temp = []
                temp.append(np.array(Image.open('/home/' + user_name + '/dataset/market1501/boundingboxtest/' + path)))
    
    
    
def get_image_path_list(train_or_test = 'train',system_user_name = 'ubuntu'):
    if train_or_test == 'train':
        folder_path = '/home/' + system_user_name + '/dataset/market1501/boundingboxtrain'
    elif train_or_test == 'test':
        folder_path = '/home/' + system_user_name + '/dataset/market1501/boundingboxtest'
    elif train_or_test == 'query':
        folder_path = '/home/' + system_user_name + '/dataset/market1501/query'
    assert os.path.isdir(folder_path)
    if train_or_test == 'train' or train_or_test == 'query':
        return sorted(os.listdir(folder_path))
    elif train_or_test == 'test':
        return sorted(os.listdir(folder_path))[6617:]
        
 
def random_select_100(user_name = 'ubuntu', num = 100):
    path_list = get_image_path_list('test',user_name)
    iden_list = sorted(np.random.choice(list(set([x[0:4] for x in path_list])),num),reverse=True)
    A = []
    B = []
    for i in xrange(len(path_list)):
        if len(iden_list) == 0:
            break
        if path_list[i][0:4] == iden_list[-1]:
            A.append(np.array(Image.open('/home/' + user_name + '/dataset/market1501/boundingboxtest/' + path_list[i])))
            j = 1
            while path_list[i][6] == path_list[i+j][6]:
                j += 1
            B.append(np.array(Image.open('/home/' + user_name + '/dataset/market1501/boundingboxtest/' + path_list[i+j])))
            iden_list.pop()
    return np.array(A)/255.,np.array(B)/255.

def get_data_for_cmc(user_name = 'ubuntu'):
    with h5py.File('test.h5','r') as f:
        A = []
        B = []
        id_list = f.keys()
        for i in id_list:
            c_list = f[i].keys()
            c1,c2 = np.random.choice(c_list,2)
            A.append(f[i][c1][np.random.randint(f[i][c1].shape[0])])
            B.append(f[i][c2][np.random.randint(f[i][c2].shape[0])])
        return np.array(A)/255.,np.array(B)/255.

   
    
if __name__ == '__main__':
    user_name = raw_input('input your system user name:')
    make_positive_index_market1501('train',user_name=user_name)
    make_positive_index_market1501('test',user_name=user_name)
