# -*- coding: utf-8 -*-
# --------------------------------------------------------
# Implementation-CVPR2015-CNN-for-ReID
# Copyright (c) 2017 Ning Ding
# Licensed under The MIT License [see LICENSE for details]
# Written by Ning Ding
# --------------------------------------------------------

"""
Model Training and Evaluation Script.
"""
import h5py
import argparse
from model import generate_model,compile_model
from data_preparation import ImageDataGenerator_for_multiinput

def main(dataset_path):
    model = generate_model()
    model = compile_model(model)
    train(model, dataset_path)

def train(model,
          h5_path,
          weights_name='weights_on_cuhk03_0_0',
          train_num=100,
          one_epoch=30000,
          epoch_num=1,
          flag_random=None,
          random_pattern=lambda x:x/2+0.4,
          flag_train=0,
          flag_val=1,
          which_val_data='validation',
          nb_val_samples=1000):
    
    with h5py.File(h5_path,'r') as f:
        Data_Generator = ImageDataGenerator_for_multiinput(width_shift_range=0.05,height_shift_range=0.05)
        Rank1s = []
        for i in xrange(train_num):
            print 'number',i,'in',train_num
            if flag_random:
                rand_x = np.random.rand()
                flag_train = random_pattern(rand_x)
            model.fit_generator(Data_Generator.flow(f,flag = flag_train),one_epoch,epoch_num,validation_data=Data_Generator.flow(f,train_or_validation=which_val_data,flag=flag_val),nb_val_samples=nb_val_samples)
            Rank1s.append(round(cmc(model)[0],2))          
            print Rank1s
            model.save_weights('weights/'+weights_name+'_'+str(i)+'.h5')
        return Rank1s

def test(model,val_or_test='test'):
    a,b = _get_test_data(val_or_test)
    return model.predict_on_batch([a,b])

def cmc(model, val_or_test='test'):
    
        a,b = _get_test_data(val_or_test)
        
        def _cmc_curve(model, camera1, camera2, rank_max=50):
            num = camera1.shape[0]    
            rank = []
            score = []    
            camera_batch1 = np.zeros(camera1.shape)
            for i in range(num):
                for j in range(num):
                    camera_batch1[j] = camera1[i]
                similarity_batch = model.predict_on_batch([camera_batch1, camera2])
                sim_trans = similarity_batch.transpose()
                similarity_rate_sorted = np.argsort(sim_trans[0])
                for k in range(num):
                    if similarity_rate_sorted[k] == i:
                        rank.append(k+1)
                        break
            rank_val = 0
            for i in range(rank_max):
                rank_val = rank_val + len([j for j in rank if i == j-1])        
                score.append(rank_val / float(num))
            return np.array(score)                
        
        return _cmc_curve(model,a,b)
        
def _get_test_data(val_or_test='test'):
    with h5py.File('cuhk-03.h5','r') as ff:    
        a = np.array([ff['a'][val_or_test][str(i)][0] for i in range(100)])
        b = np.array([ff['b'][val_or_test][str(i)][0] for i in range(100)])
        return a,b

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Main Script.')
    parser.add_argument('--dataset',
                        dest='dataset_path',
                        help='HDF5 dataset file path.',
                        required=True)
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    
    args = parse_args()
    main(args.dataset_path)





