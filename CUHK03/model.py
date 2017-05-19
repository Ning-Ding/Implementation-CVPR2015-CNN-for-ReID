# -*- coding: utf-8 -*-
# --------------------------------------------------------
# Implementation-CVPR2015-CNN-for-ReID
# Copyright (c) 2017 Ning Ding
# Licensed under The MIT License [see LICENSE for details]
# Written by Ning Ding
# --------------------------------------------------------

"""
Model Definition and Compile Script.
"""
from keras.layers import Input
from keras.layers.core import Lambda,Flatten,Dense
from keras.layers.convolutional import Conv2D,UpSampling2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import Add,Concatenate
from keras.regularizers import l2
from keras.optimizers import SGD
from keras.models import Model
from keras import backend as K

def generate_model(weight_decay=0.0005):
    '''
    define the model structure
    ---------------------------------------------------------------------------
    INPUT:
        weight_decay: all the weights in the layer would be decayed by this factor
        
    OUTPUT:
        model: the model structure after being defined
        
        # References
        - [An Improved Deep Learning Architecture for Person Re-Identification]
    ---------------------------------------------------------------------------
    '''        
    def upsample_neighbor_function(input_x):
        input_x_pad = K.spatial_2d_padding(input_x, padding=((2,2),(2,2)))
        x_length = K.int_shape(input_x)[1]
        y_length = K.int_shape(input_x)[2]
        output_x_list = []
        output_y_list = []
        for i_x in range(2, x_length + 2):
            for i_y in range(2, y_length + 2):
                output_y_list.append(input_x_pad[:,i_x-2:i_x+3,i_y-2:i_y+3,:])
            output_x_list.append(K.concatenate(output_y_list, axis=2))
            output_y_list = []
        return K.concatenate(output_x_list, axis=1)
    
    max_pooling = MaxPooling2D()
    
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
    negative = Lambda(lambda x: -x)
    x1_nn = negative(x1_nn)
    x2_nn = negative(x2_nn)
    x1 = Add()([x1_up, x2_nn])
    x2 = Add()([x2_up, x1_nn])

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
    
    y = Concatenate()([x1, x2])
    y = Flatten()(y)   
    y = Dense(500, kernel_regularizer=l2(weight_decay), activation='relu')(y)
    y = Dense(2, kernel_regularizer=l2(weight_decay), activation='softmax')(y)
    
    model = Model(inputs=[x1_input, x2_input], outputs=[y])
    model.summary()
    
    return model

def compile_model(model, *args, **kw):
    '''
    compile the model after defined
    ---------------------------------------------------------------------------
    INPUT:
        model: model before compiled
        all the other inputs should be organized as the form 
                loss='categorical_crossentropy'
        # Example
                model = compiler_def(model_def,
                                     sgd='SGD_new(lr=0.01, momentum=0.9)',
                                     loss='categorical_crossentropy',
                                     metrics='accuracy')
        # Default
                if your don't give other arguments other than model, the default
                config is the example showed above (SGD_new is the identical 
                optimizer to the one in reference paper)
    OUTPUT:
        model: model after compiled
        
        # References
        - [An Improved Deep Learning Architecture for Person Re-Identification]
    ---------------------------------------------------------------------------
    '''    
    
    class SGD_new(SGD):
        '''
        redefinition of the original SGD
        '''
        def __init__(self, lr=0.01, momentum=0., decay=0.,
                     nesterov=False, **kwargs):
            super(SGD, self).__init__(**kwargs)
            self.__dict__.update(locals())
            self.iterations = K.variable(0.)
            self.lr = K.variable(lr)
            self.momentum = K.variable(momentum)
            self.decay = K.variable(decay)
            self.inital_decay = decay
    
        def get_updates(self, params, constraints, loss):
            grads = self.get_gradients(loss, params)
            self.updates = []
    
            lr = self.lr
            if self.inital_decay > 0:
                lr *= (1. / (1. + self.decay * self.iterations)) ** 0.75
                self.updates .append(K.update_add(self.iterations, 1))
    
            # momentum
            shapes = [K.get_variable_shape(p) for p in params]
            moments = [K.zeros(shape) for shape in shapes]
            self.weights = [self.iterations] + moments
            for p, g, m in zip(params, grads, moments):
                v = self.momentum * m - lr * g  # velocity
                self.updates.append(K.update(m, v))
    
                if self.nesterov:
                    new_p = p + self.momentum * v - lr * g
                else:
                    new_p = p + v
    
                # apply constraints
                if p in constraints:
                    c = constraints[p]
                    new_p = c(new_p)
    
                self.updates.append(K.update(p, new_p))
            return self.updates 
    all_classes = {
        'sgd_new': 'SGD_new(lr=0.01, momentum=0.9)',        
        'sgd': 'SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)',
        'rmsprop': 'RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)',
        'adagrad': 'Adagrad(lr=0.01, epsilon=1e-06)',
        'adadelta': 'Adadelta(lr=1.0, rho=0.95, epsilon=1e-06)',
        'adam': 'Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)',
        'adamax': 'Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08)',
        'nadam': 'Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)',
    }
    param = {'optimizer': 'sgd_new', 'loss': 'categorical_crossentropy', 'metrics': 'accuracy'}
    config = ''
    if len(kw):    
        for (key, value) in kw.items():
            if key in param:            
                param[key] = kw[key]
            elif key in all_classes:
                config = kw[key]
            else:
                print 'error'
    if not len(config):
        config = all_classes[param['optimizer']]
    optimiz = eval(config)
    model.compile(optimizer=optimiz,
              loss=param['loss'],
              metrics=[param['metrics']])
    
    print("Model Compile Successful.")
    return model




if __name__ == "__main__":
    """
    Just for Quickly Testing.
    """
    model = generate_model()
    model = compile_model(model)
