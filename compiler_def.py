# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 17:48:09 2016

@author: CASIA
"""

import numpy as np
np.random.seed(1217)
from keras import backend as K
from keras.optimizers import SGD

def compiler_def(model, *args, **kw):
    '''
    the standard SDM training function
    ---------------------------------------------------------------------------
    INPUT:
        model: model before compiling
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
        model: model after being compiled
        
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
    return model
