# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 23:14:44 2016

@author: lenovo
"""
import numpy as np

def cmc_curve(model, camera1, camera2, rank_max=50):
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
