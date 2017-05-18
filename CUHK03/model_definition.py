# -*- coding: utf-8 -*-
# --------------------------------------------------------
# Implementation-CVPR2015-CNN-for-ReID
# Copyright (c) 2017 Ning Ding
# Licensed under The MIT License [see LICENSE for details]
# Written by Ning Ding
# --------------------------------------------------------

"""
Model Definition Script.
"""
from __future__ import absolute_import

import tensorflow as tf
from tensorflow.contrib.keras.python.keras.layers.convolutional import Conv2D
from tensorflow.contrib.keras.python.keras.layers.pooling import MaxPooling2D
from tensorflow.contrib.keras.python.keras.regularizers import l2


def tf_model_definition(weight_decay=0.0005):

    x1 = tf.placeholder(tf.float32, shape=(-1, 160, 60, 3), name="input_x1")
    x2 = tf.placeholder(tf.float32, shape=(-1, 160, 60, 3), name="input_x2")
    share_conv_1 = Conv2D(20, 5, kernel_regularizer=l2(weight_decay), activation="relu")
    x1 = share_conv_1(x1)
    x2 = share_conv_1(x2)
    x1 = MaxPooling2D(x1)
    x2 = MaxPooling2D(x2)
    share_conv_2 = Conv2D(25, 5, kernel_regularizer=l2(weight_decay), activation="relu")
    x1 = share_conv_2(x1)
    x2 = share_conv_2(x2)
    x1 = MaxPooling2D(x1)
    x2 = MaxPooling2D(x2)

    # a8 = merge([a7,b7],mode=cross_input_asym,output_shape=cross_input_shape)
    # b8 = merge([b7,a7],mode=cross_input_asym,output_shape=cross_input_shape)
    # a9 = Convolution2D(25,5,5, subsample=(5,5), dim_ordering='tf',activation='relu', W_regularizer=l2(l=weight_decay))(a8)
    # b9 = Convolution2D(25,5,5, subsample=(5,5), dim_ordering='tf',activation='relu', W_regularizer=l2(l=weight_decay))(b8)
    # a10 = Convolution2D(25,3,3, subsample=(1,1), dim_ordering='tf',activation='relu', W_regularizer=l2(l=weight_decay))(a9)
    # b10 = Convolution2D(25,3,3, subsample=(1,1), dim_ordering='tf',activation='relu', W_regularizer=l2(l=weight_decay))(b9)
    # a11 = MaxPooling2D((2,2),dim_ordering='tf')(a10)
    # b11 = MaxPooling2D((2,2),dim_ordering='tf')(b10)
    # c1 = merge([a11, b11], mode='concat', concat_axis=-1)
    # c2 = Flatten()(c1)
    # c3 = Dense(500,activation='relu', W_regularizer=l2(l=weight_decay))(c2)
    # c4 = Dense(2,activation='softmax', W_regularizer=l2(l=weight_decay))(c3)
    
    # model = Model(input=[a1,b1],output=c4)
    # model.summary()
    
    # print 'model definition complete'
    # return model

    # '''  
    # K._IMAGE_DIM_ORDERING = 'tf'    
    # def concat_iterat(input_tensor):
    #     input_expand = K.expand_dims(K.expand_dims(input_tensor, -2), -2)
    #     x_axis = []
    #     y_axis = []
    #     for x_i in range(5):
    #         for y_i in range(5):
    #             y_axis.append(input_expand)
    #         x_axis.append(K.concatenate(y_axis, axis=2))
    #         y_axis = []
    #     return K.concatenate(x_axis, axis=1)
            
    # def cross_input_asym(X):
    #     tensor_left = X[0]
    #     tensor_right = X[1]
    #     x_length = K.int_shape(tensor_left)[1]
    #     y_length = K.int_shape(tensor_left)[2]
    #     cross_y = []
    #     cross_x = []
    #     tensor_left_padding = K.spatial_2d_padding(tensor_left,padding=(2,2))
    #     tensor_right_padding = K.spatial_2d_padding(tensor_right,padding=(2,2))
    #     for i_x in range(2, x_length + 2):
    #         for i_y in range(2, y_length + 2):
    #             cross_y.append(tensor_left_padding[:,i_x-2:i_x+3,i_y-2:i_y+3,:] 
    #                          - concat_iterat(tensor_right_padding[:,i_x,i_y,:]))
    #         cross_x.append(K.concatenate(cross_y,axis=2))
    #         cross_y = []
    #     cross_out = K.concatenate(cross_x,axis=1)
    #     return K.abs(cross_out)
        
    # def cross_input_shape(input_shapes):
    #     input_shape = input_shapes[0]
    #     return (input_shape[0],input_shape[1] * 5,input_shape[2] * 5,input_shape[3])
