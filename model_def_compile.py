# -*- coding: utf-8 -*-
import numpy as np
np.random.seed(1217)
import h5py
from keras import backend as K
from keras.models import Model
from keras.layers import Input,Dense,Convolution2D,Activation,MaxPooling2D,Flatten,merge
from keras.regularizers import l2
from keras.optimizers import SGD,RMSprop,Adagrad,Adadelta,Adam,Adamax,Nadam
from keras.preprocessing import image as pre_image

def model_def(flag=0, weight_decay=0.0005):
    '''
    define the model structure
    ---------------------------------------------------------------------------
    INPUT:
        flag: used to decide which model structure you want to use
                the default value is 0, which refers to the same structure as paper in the reference
        weight_decay: all the weights in the layer would be decayed by this factor
        
    OUTPUT:
        model: the model structure after being defined
        
        # References
        - [An Improved Deep Learning Architecture for Person Re-Identification]
    ---------------------------------------------------------------------------
    '''        
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
    

def compiler_def(model, *args, **kw):
    '''
    compile the model after definited
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



class NumpyArrayIterator_for_multiinput_from_hdf5(pre_image.Iterator):

    def __init__(self, f, train_or_validation = 'train', image_data_generator=None,
                 batch_size=32, shuffle=False, seed=None,
                 dim_ordering='default'):
        
        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        self.f = f
        self.train_or_validation = train_or_validation
        self.image_data_generator = image_data_generator
        self.dim_ordering = dim_ordering
        super(NumpyArrayIterator_for_multiinput_from_hdf5, self).__init__(f[train_or_validation]['y'].shape[0], batch_size, shuffle, seed)

    def next(self):
        # for python 2.x.
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch
        # see http://anandology.com/blog/using-iterators-and-generators/
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        # The transformation of images is not under thread lock so it can be done in parallel
        batch_x1 = np.zeros(tuple([current_batch_size] + list(self.f[self.train_or_validation]['x1'].shape[1:])))
        batch_x2 = np.zeros(tuple([current_batch_size] + list(self.f[self.train_or_validation]['x2'].shape[1:])))
        for i, j in enumerate(index_array):
            x1 = self.f[self.train_or_validation]['x1'][j]
            x2 = self.f[self.train_or_validation]['x2'][j]
            if self.image_data_generator is None:
                batch_x1[i] = x1
                batch_x2[i] = x2
                continue
            else:
                x1 = self.image_data_generator.random_transform(x1.astype('float32'))
                x2 = self.image_data_generator.random_transform(x2.astype('float32'))
                x1 = self.image_data_generator.standardize(x1)
                x2 = self.image_data_generator.standardize(x2)
                batch_x1[i] = x1
                batch_x2[i] = x2
       
        y = self.f[self.train_or_validation]['y'][:]
        batch_y = y[index_array]
        return [batch_x1,batch_x2], batch_y


if __name__ == '__main__':
    model = model_def()
    print 'model definition done.'
    model = compiler_def(model)
    print 'model compile done.'
    f = h5py.File('cuhk-03_for_CNN.h5','r')
    #val_data = [[f['validation']['x1'][:],f['validation']['x2'][:]],f['validation']['y'][:]]
    print 'validation data loaded.'
    f_iter = NumpyArrayIterator_for_multiinput_from_hdf5(f,batch_size=150)
    f_iter_val = NumpyArrayIterator_for_multiinput_from_hdf5(f,train_or_validation='validation',batch_size=1000)
    print 'begin to fit!'
    model.fit_generator(f_iter,150000,10,validation_data=f_iter_val)

