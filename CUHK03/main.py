# -*- coding: utf-8 -*-
# --------------------------------------------------------
# Implementation-CVPR2015-CNN-for-ReID
# Copyright (c) 2017 Ning Ding
# Licensed under The MIT License [see LICENSE for details]
# Written by Ning Ding
# --------------------------------------------------------

import os
import sys
import h5py
import itertools
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2
from easydict import EasyDict

__C = EasyDict()
__C.DATA = EasyDict()
__C.DATA.ORIGINAL_FILE = "cuhk-03.mat"
__C.DATA.CREATED_FILE = "cuhk-03.hdf5"
__C.DATA.INDEX_FILE = "cuhk-03-index.hdf5"
__C.DATA.IMAGE_SIZE = (60, 160)
__C.DATA.ARRAY_SIZE = (160, 60)
__C.DATA.PATTERN = EasyDict()
__C.DATA.PATTERN.TRAIN = [1, 0, 0]
__C.DATA.PATTERN.VALID = [1, 0]
__C.TRAIN = EasyDict()
__C.TRAIN.BATCHSIZE = 150
__C.TRAIN.STEPS = 2100
__C.TRAIN.WEIGHT_DECAY = 0.00025
__C.TRAIN.GPU_INDEX = 0
__C.VALID = EasyDict()
__C.VALID.STEPS = 1

cfg = __C

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.TRAIN.GPU_INDEX)


# -------------------------------------------------------
# -------------------------------------------------------
# model section
# @export_estimator
def generate_model(weight_decay=cfg.TRAIN.WEIGHT_DECAY):

    # Input Pair of images
    x1_input = keras.layers.Input(shape=(*cfg.DATA.ARRAY_SIZE, 3),
                                  name="x1_input")

    x2_input = keras.layers.Input(shape=(*cfg.DATA.ARRAY_SIZE, 3),
                                  name="x2_input")

    # Tied Convolution with max pooling
    share_conv_1 = keras.layers.Conv2D(20, 5,
                                       activation="relu",
                                       kernel_regularizer=l2(weight_decay),
                                       name="share_conv_1")

    x1 = share_conv_1(x1_input)
    x2 = share_conv_1(x2_input)
    x1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(x1)
    x2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(x2)

    share_conv_2 = keras.layers.Conv2D(25, 5,
                                       activation="relu",
                                       kernel_regularizer=l2(weight_decay),
                                       name="share_conv_2")

    x1 = share_conv_2(x1)
    x2 = share_conv_2(x2)
    x1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(x1)
    x2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(x2)

    # Cross-Input Neighborhood Differences
    x1_up = keras.layers.UpSampling2D(size=(5, 5))(x1)
    x2_up = keras.layers.UpSampling2D(size=(5, 5))(x2)

    x1_nn = keras.layers.Lambda(_upsample_neighbor_function)(x1)
    x2_nn = keras.layers.Lambda(_upsample_neighbor_function)(x2)

    x1_nn = keras.layers.Lambda(lambda x: -x)(x1_nn)
    x2_nn = keras.layers.Lambda(lambda x: -x)(x2_nn)

    x1 = keras.layers.Add()([x1_up, x2_nn])
    x2 = keras.layers.Add()([x2_up, x1_nn])

    # Patch Summary Features
    conv_3_1 = keras.layers.Conv2D(25, 5, strides=(5, 5),
                                   activation="relu",
                                   kernel_regularizer=l2(weight_decay),
                                   name="conv_3_1")

    conv_3_2 = keras.layers.Conv2D(25, 5, strides=(5, 5),
                                   activation="relu",
                                   kernel_regularizer=l2(weight_decay),
                                   name="conv_3_2")
    x1 = conv_3_1(x1)
    x2 = conv_3_2(x2)

    # Across-Patch Features
    conv_4_1 = keras.layers.Conv2D(25, 3,
                                   activation="relu",
                                   kernel_regularizer=l2(weight_decay),
                                   name="conv_4_1")

    conv_4_2 = keras.layers.Conv2D(25, 3,
                                   activation="relu",
                                   kernel_regularizer=l2(weight_decay),
                                   name="conv_4_2")
    x1 = conv_4_1(x1)
    x2 = conv_4_2(x2)
    x1 = keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x1)
    x2 = keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x2)

    # Higher-Order Relationships
    y = keras.layers.Concatenate()([x1, x2])
    y = keras.layers.Flatten()(y)
    y = keras.layers.Dense(500,
                           kernel_regularizer=l2(weight_decay),
                           activation='relu')(y)

    y = keras.layers.Dense(2,
                           kernel_regularizer=l2(weight_decay),
                           activation='softmax')(y)

    model = keras.Model(inputs=[x1_input, x2_input], outputs=[y])
    model.summary()

    model = _compile_model(model)

    return model


def _upsample_neighbor_function(input_x):
    input_x_pad = K.spatial_2d_padding(input_x, padding=((2, 2), (2, 2)))
    x_length = K.int_shape(input_x)[1]
    y_length = K.int_shape(input_x)[2]
    output_x_list = []
    output_y_list = []
    for i_x in range(2, x_length + 2):
        for i_y in range(2, y_length + 2):
            output_y_list.append(input_x_pad[:, i_x-2:i_x+3, i_y-2:i_y+3, :])
        output_x_list.append(K.concatenate(output_y_list, axis=2))
        output_y_list = []
    return K.concatenate(output_x_list, axis=1)


def _compile_model(model):
    sgd = tf.keras.optimizers.SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer=sgd,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


# -------------------------------------------------------
# dataset preparation section

def get_data_generator(mode='train', pattern=cfg.DATA.PATTERN.TRAIN):
    def gen_data():
        for pos_or_neg in itertools.cycle(pattern):
            if pos_or_neg:
                image_x, image_y = _generate_positive_pair(mode)
                yield ((image_x, image_y), (1, 0))
            else:
                image_x, image_y = _generate_negative_pair(mode)
                yield ((image_x, image_y), (0, 1))

    return gen_data


def _generate_positive_pair(mode='train'):
    with h5py.File(cfg.DATA.CREATED_FILE, 'r') as f:
        index_array = _get_index_array(mode)
        i = np.random.choice(index_array)
        x, y = np.random.choice(f[str(i)].shape[0], 2, replace=False)
        image_x = _image_augmentation(f[str(i)][x])
        image_y = _image_augmentation(f[str(i)][y])
        return image_x, image_y


def _generate_negative_pair(mode='train'):
    with h5py.File(cfg.DATA.CREATED_FILE, 'r') as f:
        index_array = _get_index_array(mode)
        i, j = np.random.choice(index_array, 2, replace=False)
        x = np.random.choice(f[str(i)].shape[0], replace=False)
        y = np.random.choice(f[str(j)].shape[0], replace=False)
        image_x = f[str(i)][x]
        image_y = f[str(j)][y]
        return image_x, image_y


def _get_index_array(mode='train'):

    with h5py.File(cfg.DATA.INDEX_FILE, 'r') as f:
        index_array = f[mode][:]

    return index_array


# soon
def _image_augmentation(image):
    x_padding = int(np.round(image.shape[0] * 0.05))
    y_padding = int(np.round(image.shape[1] * 0.05))
    padding_shape = x_padding * 2, y_padding * 2
    image_shape = np.array(image.shape[:2]) + np.array(padding_shape)
    image_padding = np.zeros((image_shape[0]+padding_shape[0],
                              image_shape[1]+padding_shape[1],
                              3))
    image_padding[x_padding:x_padding+image.shape[0],
                  y_padding:y_padding+image.shape[1],
                  :] = image

    x_translation = np.random.choice(x_padding * 2)
    y_translation = np.random.choice(y_padding * 2)
    new_image = image_padding[x_translation:x_translation+image.shape[0],
                              y_translation:y_translation+image.shape[1],
                              :]

    return new_image


# -------------------------------------------------------
# dataset creation section

def generate_data():

    with h5py.File(cfg.DATA.ORIGINAL_FILE, 'r') as fr, h5py.File(cfg.DATA.CREATED_FILE, 'w') as fw:
        """f[f[f['labeled'][0][i]][j][k]] get a HDF5 Dataset
        i: index from 0-4 denoted different cameras.
        j: index from 0-9 denoted different photos of identities.
        k: index denoted different identities captured by the camera.

        notes:
        1. not all j range of 0-9, it may only range 0-5.
        2. the numpy arrays are with different size.
        3. the numpy array's dimenstion: channels, width, height.
        4. there are 1360 identities from index i 0-2.
        5. the five camera each contain: 843, 440, 77, 58, 49 identities.
        """
        for i in range(3):
            for k in range(_get_identity_size(fr, i)):
                print("Now generated {} identities.".format(_compute_index(i, k) + 1))
                temp = []
                for j in range(10):
                    array = _get_array(fr, i, j, k)
                    if array is not None:
                        temp.append(array)

                fw.create_dataset(str(_compute_index(i, k)), data=np.array(temp))

        print("HDF5 Dataset Already Created.")


def random_split_dataset():

    index_test = np.random.choice(1360, 100, replace=False)
    res = np.array(list((set(range(1360)) - set(index_test))))
    index_valid = np.random.choice(res, 100, replace=False)
    index_train = np.array(list((set(res) - set(index_valid))))

    with h5py.File(cfg.DATA.INDEX_FILE, 'w') as f:
        f.create_dataset('train', data=index_train)
        f.create_dataset('valid', data=index_valid)
        f.create_dataset('test', data=index_test)

    print("Index Dataset Already Created.")


def image_preprocessing(transpose=(2, 1, 0), image_size=(60, 160)):
    def image_preprocessing_decorator(fn):
        def updated_fn(*args, **kw):
            result = fn(*args, **kw)
            result = result if len(result.shape) == 3 else None
            if result is not None:
                image = Image.fromarray(result[:].transpose(transpose))
                image = image.resize(image_size)

                # default return (160,60,3) 0-1 dtype=float64 numpy array.
                return np.array(image) / 255.
            else:
                return None
        return updated_fn
    return image_preprocessing_decorator


@image_preprocessing(image_size=cfg.DATA.IMAGE_SIZE)
def _get_array(File, camera, num, identities):
    return File[File[File['labeled'][0][camera]][num][identities]]


def _get_identity_size(File, camera):
    return File[File['labeled'][0][camera]][0].size


def _compute_index(i, k):
    if i == 0:
        return k
    elif i == 1:
        return k+843
    elif i == 2:
        return k+843+440


# -------------------------------------------------------
# training section

def train_input_fn():
    dataset = tf.data.Dataset.from_generator(
                     get_data_generator(),
                     ((tf.float32, tf.float32), tf.int8),
                     ((tf.TensorShape([*cfg.DATA.ARRAY_SIZE, 3]),
                       tf.TensorShape([*cfg.DATA.ARRAY_SIZE, 3])),
                      tf.TensorShape(None))
                    )

    dataset = dataset.batch(batch_size=cfg.TRAIN.BATCHSIZE)
    dataset = dataset.prefetch(buffer_size=150)

    return dataset


def valid_input_fn():
    dataset = tf.data.Dataset.from_generator(
        get_data_generator(mode='valid', pattern=cfg.DATA.PATTERN.VALID),
        ((tf.float32, tf.float32), tf.int8),
        ((tf.TensorShape([*cfg.DATA.ARRAY_SIZE, 3]),
          tf.TensorShape([*cfg.DATA.ARRAY_SIZE, 3])),
         tf.TensorShape(None)))

    dataset = dataset.batch(100)
    dataset = dataset.prefetch(buffer_size=100)

    return dataset


def prepare_keras_callback():
    callbacks = []

    callback_lrs = tf.keras.callbacks.LearningRateScheduler(
                schedule=_learning_rate_schedule,
                verbose=1)
    callbacks.append(callback_lrs)

    callback_tensorboard = tf.keras.callbacks.TensorBoard(
                log_dir='./logs',
                histogram_freq=0,
                write_graph=True,
                write_grads=True,
                write_images=True)
    callbacks.append(callback_tensorboard)

    callback_mcp = tf.keras.callbacks.ModelCheckpoint(
                filepath='weights.{epoch:06d}-{val_loss:.4f}.checkpoint',
                monitor='val_loss',
                verbose=1,
                mode='auto',
                period=1000)
    callbacks.append(callback_mcp)

    return callbacks


def _learning_rate_schedule(epoch):
    step = epoch * 100
    learning_rate = 0.01 * (1 + 0.0001 * step) ** (-0.75)
    return learning_rate


# -------------------------------------------------------
# decorator of turning keras model to tensorflow estimator.
def export_estimator(fn):
    def new_fn(*arg, **kw):
        model = fn(*arg, **kw)
        estimator = tf.keras.estimator.model_to_estimator(model)
        return estimator
    return new_fn


# -------------------------------------------------------
# file check
def dataset_file_check():
    _check_created_dataset()
    _check_index_dataset()


def _check_created_dataset():
    if not os.path.exists(cfg.DATA.CREATED_FILE):
        print("Can't find the created HDF5 dataset file.")
        print("Would you like to Create a new one?")
        cmd = input("yes(y) or no(n)?")
        if cmd == 'y':
            _check_original_dataset()
            generate_data()
        else:
            sys.exit()


def _check_original_dataset():
    if not os.path.exists(cfg.DATA.ORIGINAL_FILE):
        print("Can't find the original dataset file.")
        print("Usually named: 'cuhk-03.mat'.")
        print("Find it, and come back lator.")
        sys.exit()


def _check_index_dataset():
    if not os.path.exists(cfg.DATA.INDEX_FILE):
        print("File is not Exists.")
        print("Creating new index file.")
        random_split_dataset()


if __name__ == '__main__':

    dataset_file_check()

    model = generate_model()
    train_dataset = train_input_fn()
    valid_dataset = valid_input_fn()
    callbacks = prepare_keras_callback()

    history = model.fit(train_dataset,
                        validation_data=valid_dataset,
                        callbacks=callbacks,
                        verbose=1,
                        steps_per_epoch=100,
                        epochs=cfg.TRAIN.STEPS,
                        validation_steps=1)

    # estimator = generate_model()
    # estimator.train(steps=cfg.TRAIN.STEPS, input_fn=train_input_fn)
