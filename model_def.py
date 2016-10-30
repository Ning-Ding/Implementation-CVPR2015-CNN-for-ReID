# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 17:24:18 2016

@author: CASIA
"""

import numpy as np
np.random.seed(1217)

from keras import backend as K
from keras.models import Model
from keras.layers import Input,Dense,Convolution2D,Activation,MaxPooling2D,Flatten,merge
from keras.regularizers import l2

def model_def(flag=0, weight_decay=0.0005):
    K._IMAGE_DIM_ORDERING = 'tf'    
    def concat_iterat(input_tensor):
        input_expand = K.expand_dims(K.expand_dims(input_tensor, -2), -2)
        x_axis = []
        y_axis = []
        for x_i in range(5):
            for y_i in range(5):
                y_axis.append(input_expand)
            x_axis.append(K.concatenate(y_axis, axis=2))
            y_axis = []
        return K.concatenate(x_axis, axis=1)
    
    def cross_input_sym(X):
        tensor_left = X[0]
        tensor_right = X[1]
        x_length = K.int_shape(tensor_left)[1]
        y_length = K.int_shape(tensor_left)[2]
        cross_y = []
        cross_x = []
        tensor_left_padding = K.spatial_2d_padding(tensor_left,padding=(2,2))
        tensor_right_padding = K.spatial_2d_padding(tensor_right,padding=(2,2))
        for i_x in range(2, x_length + 2):
            for i_y in range(2, y_length + 2):
                cross_y.append(tensor_left_padding[:,i_x-2:i_x+3,i_y-2:i_y+3,:] 
                             - tensor_right_padding[:,i_x-2:i_x+3,i_y-2:i_y+3,:])
            cross_x.append(K.concatenate(cross_y,axis=2))
            cross_y = []
        cross_out = K.concatenate(cross_x,axis=1)
        return K.abs(cross_out)
            
    def cross_input_asym(X):
        tensor_left = X[0]
        tensor_right = X[1]
        x_length = K.int_shape(tensor_left)[1]
        y_length = K.int_shape(tensor_left)[2]
        cross_y = []
        cross_x = []
        tensor_left_padding = K.spatial_2d_padding(tensor_left,padding=(2,2))
        tensor_right_padding = K.spatial_2d_padding(tensor_right,padding=(2,2))
        for i_x in range(2, x_length + 2):
            for i_y in range(2, y_length + 2):
                cross_y.append(tensor_left_padding[:,i_x-2:i_x+3,i_y-2:i_y+3,:] 
                             - concat_iterat(tensor_right_padding[:,i_x,i_y,:]))
            cross_x.append(K.concatenate(cross_y,axis=2))
            cross_y = []
        cross_out = K.concatenate(cross_x,axis=1)
        return K.abs(cross_out)
        
    def cross_input_shape(input_shapes):
        input_shape = input_shapes[0]
        return (input_shape[0],input_shape[1] * 5,input_shape[2] * 5,input_shape[3])
        
    '''
    model definition begin
    -------------------------------------------------------------------------------
    '''
    if flag == 0:
        print 'now begin to compile the model with the difference between ones and neighbour matrixs.'
        
        a1 = Input(shape=(160,60,3))
        b1 = Input(shape=(160,60,3))
        share = Convolution2D(20,5,5,dim_ordering='tf', W_regularizer=l2(l=weight_decay))
        a2 = share(a1)
        b2 = share(b1)
        a3 = Activation('relu')(a2)
        b3 = Activation('relu')(b2)
        a4 = MaxPooling2D(dim_ordering='tf')(a3)
        b4 = MaxPooling2D(dim_ordering='tf')(b3)
        share2 = Convolution2D(25,5,5,dim_ordering='tf', W_regularizer=l2(l=weight_decay))
        a5 = share2(a4)
        b5 = share2(b4)
        a6 = Activation('relu')(a5)
        b6 = Activation('relu')(b5)
        a7 = MaxPooling2D(dim_ordering='tf')(a6)
        b7 = MaxPooling2D(dim_ordering='tf')(b6)
        a8 = merge([a7,b7],mode=cross_input_asym,output_shape=cross_input_shape)
        b8 = merge([b7,a7],mode=cross_input_asym,output_shape=cross_input_shape)
        a9 = Convolution2D(25,5,5, subsample=(5,5), dim_ordering='tf',activation='relu', W_regularizer=l2(l=weight_decay))(a8)
        b9 = Convolution2D(25,5,5, subsample=(5,5), dim_ordering='tf',activation='relu', W_regularizer=l2(l=weight_decay))(b8)
        a10 = Convolution2D(25,3,3, subsample=(1,1), dim_ordering='tf',activation='relu', W_regularizer=l2(l=weight_decay))(a9)
        b10 = Convolution2D(25,3,3, subsample=(1,1), dim_ordering='tf',activation='relu', W_regularizer=l2(l=weight_decay))(b9)
        a11 = MaxPooling2D((2,2),dim_ordering='tf')(a10)
        b11 = MaxPooling2D((2,2),dim_ordering='tf')(b10)
        c1 = merge([a11, b11], mode='concat', concat_axis=-1)
        c2 = Flatten()(c1)
        c3 = Dense(500,activation='relu', W_regularizer=l2(l=weight_decay))(c2)
        c4 = Dense(2,activation='softmax', W_regularizer=l2(l=weight_decay))(c3)
        
        model = Model(input=[a1,b1],output=c4)
        model.summary()
        
    if flag == 1:
        print 'now begin to compile the model with the difference between both neighbour matrixs.'
        
        a1 = Input(shape=(160,60,3))
        b1 = Input(shape=(160,60,3))
        share = Convolution2D(20,5,5,dim_ordering='tf', W_regularizer=l2(l=weight_decay))
        a2 = share(a1)
        b2 = share(b1)
        a3 = Activation('relu')(a2)
        b3 = Activation('relu')(b2)
        a4 = MaxPooling2D(dim_ordering='tf')(a3)
        b4 = MaxPooling2D(dim_ordering='tf')(b3)
        share2 = Convolution2D(25,5,5,dim_ordering='tf', W_regularizer=l2(l=weight_decay))
        a5 = share2(a4)
        b5 = share2(b4)
        a6 = Activation('relu')(a5)
        b6 = Activation('relu')(b5)
        a7 = MaxPooling2D(dim_ordering='tf')(a6)
        b7 = MaxPooling2D(dim_ordering='tf')(b6)
        c1 = merge([a7,b7],mode=cross_input_sym,output_shape=cross_input_shape)
        c2 = Convolution2D(25,5,5, subsample=(5,5), dim_ordering='tf',activation='relu', W_regularizer=l2(l=weight_decay))(c1)
        c3 = Convolution2D(25,3,3, subsample=(1,1), dim_ordering='tf',activation='relu', W_regularizer=l2(l=weight_decay))(c2)
        c4 = MaxPooling2D((2,2),dim_ordering='tf')(c3)
        c5 = Flatten()(c4)
        c6 = Dense(10,activation='relu', W_regularizer=l2(l=weight_decay))(c5)
        c7 = Dense(2,activation='softmax', W_regularizer=l2(l=weight_decay))(c6)
        
        model = Model(input=[a1,b1],output=c7)
        model.summary()
    
    print 'model definition complete'
    return model