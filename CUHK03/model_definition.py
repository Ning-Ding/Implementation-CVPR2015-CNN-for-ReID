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
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.keras.python.keras.layers.convolutional import Conv2D,UpSampling2D
from tensorflow.contrib.keras.python.keras.layers.pooling import MaxPool2D
from tensorflow.contrib.keras.python.keras.regularizers import l2
from tensorflow.contrib.keras.python.keras.layers.core import Lambda,Flatten,Dense
from tensorflow.contrib.keras.python.keras.engine.topology import Input


def tf_model_definition(weight_decay=0.0005):

    def upsample_neighbor_function(X):
        input_tensor_pad = tf.pad(X,[[0,0],[2,2],[2,2],[0,0]])
        x_length = tf.shape(X)[1]
        y_length = tf.shape(X)[2]
        output_x_list = []
        output_y_list = []
        for i_x in range(2, x_length + 2):
            for i_y in range(2, y_length + 2):
                output_y_list.append(input_tensor_pad[:,i_x-2:i_x+3,i_y-2:i_y+3,:])
            output_x_list.append(tf.concat(output_y_list, axis=2))
            output_y_list = []
        return tf.concat(output_x_list, axis=1)
    
    def upsample_neighbor_shape(input_shape):
        return (input_shape[0],input_shape[1] * 5,input_shape[2] * 5,input_shape[3])
    
    max_pooling = MaxPool2D(2)
    
    x1_input = Input(shape=(160,60,3))
    x2_input = Input(shape=(160,60,3))
    
    share_conv_1 = Conv2D(20, 5, kernel_regularizer=l2(weight_decay), activation="relu")
    x1 = share_conv_1(x1_input)
    x2 = share_conv_1(x2_input)
    x1 = max_pooling(x1)
    x2 = max_pooling(x2)
    
    share_conv_2 = Conv2D(25, 5, kernel_regularizer=l2(weight_decay), activation="relu")
    x1 = share_conv_2(x1)
    x2 = share_conv_2(x2)
    x1 = max_pooling(x1)
    x2 = max_pooling(x2)
    
    upsample_same = UpSampling2D(size=(5, 5))
    x1_up = upsample_same(x1)
    x2_up = upsample_same(x2)    
    upsample_neighbor = Lambda(upsample_neighbor_function)        
    x1_nn = upsample_neighbor(x1)
    x2_nn = upsample_neighbor(x2)
    
    x1 = tf.add(x1_up, tf.negative(x2_nn))
    x2 = tf.add(x2_up, tf.negative(x1_nn))    
    
    conv_3_1 = Conv2D(25, 5, strides=(5, 5), kernel_regularizer=l2(weight_decay), activation="relu")
    conv_3_2 = Conv2D(25, 5, strides=(5, 5), kernel_regularizer=l2(weight_decay), activation="relu")
    x1 = conv_3_1(x1)
    x2 = conv_3_2(x2)
    
    conv_4_1 = Conv2D(25, 3, kernel_regularizer=l2(weight_decay), activation="relu")
    conv_4_2 = Conv2D(25, 3, kernel_regularizer=l2(weight_decay), activation="relu")
    x1 = conv_4_1(x1)
    x2 = conv_4_2(x2)
    x1 = max_pooling(x1)
    x2 = max_pooling(x2)
    
    y = tf.concat([x1, x2], -1)
    y = Flatten(y)
    
    FC_1 = Dense(500, kernel_regularizer=l2(weight_decay), activation='relu')
    FC_2 = Dense(2, kernel_regularizer=l2(weight_decay), activation='softmax')
    y = FC_1(y)
    y_output = FC_2(y)
    
    model = Model(input=[x1_input, x2_input],output=y)
    model.summary()
    
    return model


if __name__ == "__main__":
    """
    Just for Quickly Testing.
    """
    model = tf_model_definition()
