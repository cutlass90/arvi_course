# -*- coding: utf-8 -*-
"""ResNet50 model for Keras.

# Reference:

- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

Adapted from code contributed by BigMoyan.
"""
from __future__ import print_function
from __future__ import absolute_import

import warnings

from keras.layers import Input
from keras import layers
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Conv1D
from keras.layers import AveragePooling1D
from keras.layers import GlobalAveragePooling1D
from keras.layers import BatchNormalization, Reshape, Concatenate
from keras.models import Model
from keras import backend as K
from keras.engine.topology import get_source_inputs
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.imagenet_utils import _obtain_input_shape

from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Conv2DTranspose, Lambda
import tensorflow as tf

from tools import Conv1DTranspose




def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    if input_tensor.get_shape().as_list()[-1] != filters3:
        # print('convert ', input_tensor.get_shape().as_list()[-1], 'to', filters3)
        input_tensor = Conv1D(filters3, 1, name=conv_name_base + 'conv')(input_tensor)
        input_tensor = BatchNormalization(name=bn_name_base + 'conv')(input_tensor)
        input_tensor = LeakyReLU(alpha=0.3)(input_tensor)


    x = Conv1D(filters1, 1, name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = LeakyReLU(alpha=0.3)(x)

    x = Conv1D(filters2, kernel_size,
               padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = LeakyReLU(alpha=0.3)(x)

    x = Conv1D(filters3, 1, name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = LeakyReLU(alpha=0.3)(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=2):
    """A block that has a conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.

    """
    filters1, filters2, filters3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv1D(filters1, 1, strides=strides,
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = LeakyReLU(alpha=0.3)(x)

    x = Conv1D(filters2, kernel_size, padding='same',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = LeakyReLU(alpha=0.3)(x)

    x = Conv1D(filters3, 1, name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    shortcut = Conv1D(filters3, 1, strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = LeakyReLU(alpha=0.3)(x)
    return x


def deconv_block(input_tensor, kernel_size, filters, stage, block, strides=2):
    """A block that has a deconv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv1DTranspose(inputs=input_tensor,
                        filters=filters1,
                        kernel_size=1,
                        strides=strides,
                        padding='same')
               
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = LeakyReLU(alpha=0.3)(x)

    x = Conv1DTranspose(inputs=x,
                        filters=filters2,
                        kernel_size=kernel_size,
                        strides=1,
                        padding='same')
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = LeakyReLU(alpha=0.3)(x)

    x = Conv1DTranspose(inputs=x,
                        filters=filters3,
                        kernel_size=kernel_size,
                        strides=1,
                        padding='same')
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    shortcut = Conv1DTranspose(inputs=input_tensor,
                        filters=filters3,
                        kernel_size=kernel_size,
                        strides=strides,
                        padding='same')
    shortcut = BatchNormalization(name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = LeakyReLU(alpha=0.3)(x)
    return x

def ResNetDiscriminator(c):
    """
    Returns
        A Keras model instance.
    """
    print('#'*10, ' Create discriminator ', '#'*10)
    DICT_SIZE = len(c.char_to_class)
    signal_input = Input(shape=(c.audio_size,))
    x = Reshape([c.audio_size, 1])(signal_input)
    x = Conv1D(
        64, 7, strides=2, padding='same', name='conv1')(x)
    x = BatchNormalization(name='bn_conv1')(x)
    x = LeakyReLU(alpha=0.3)(x)
    for i in range(1, c.n_compress_block+1):
        x = conv_block(x, 3,
                       [i**2*c.convo_size, i**2*c.convo_size, 4*i**2*c.convo_size],
                       stage=i, block='a')
        x = identity_block(x, 3,
                           [i**2*c.convo_size, i**2*c.convo_size, 4*i**2*c.convo_size],
                           stage=i, block='b')
        x = identity_block(x, 3,
                           [i**2*c.convo_size, i**2*c.convo_size, 4*i**2*c.convo_size],
                           stage=i, block='c')
    print('x', x)
    
    out_chars = conv_block(x, 3,
                   [8**2*c.convo_size, 8**2*c.convo_size, 4*8**2*c.convo_size],
                   stage=42, block='a')
    out_chars = conv_block(out_chars, 3,
                   [7**2*c.convo_size, 7**2*c.convo_size, 4*7**2*c.convo_size],
                   stage=43, block='a')
    out_chars = Conv1D(DICT_SIZE, 1, padding='same')(out_chars)
    out_chars = Activation('softmax', name='out_chars')(out_chars)
    print('out_chars', out_chars)

    true_fake_many = identity_block(x, 3,
                           [256, 256, 4*256],
                           stage=44, block='c')
    true_fake_many = Conv1D(1, 1)(true_fake_many)
    true_fake_many = Reshape([256,])(true_fake_many)
    true_fake_many = Activation('sigmoid', name='true_fake_many')(true_fake_many)
    print('true_fake_many', true_fake_many)

    true_fake_1 = Conv1D(1024, 3, strides=1)(x)
    true_fake_1 = Conv1D(1, 1)(true_fake_1)
    true_fake_1 = GlobalAveragePooling1D()(true_fake_1)
    true_fake_1 = Activation('sigmoid', name='true_fake_1')(true_fake_1)
    print('true_fake_1', true_fake_1)
    # Create model.
    model = Model(inputs=signal_input, outputs=[out_chars,
                                                true_fake_many,
                                                true_fake_1])

    return model


def ResNetGenerator(c):
    """
    Returns
        A Keras model instance.
    """
    print('#'*10, ' Create generator ', '#'*10)
    net = {}
    signal_input = Input(shape=(c.audio_size,))
    x = Reshape([c.audio_size, 1])(signal_input)


    # COMPRESS
    x = Conv1D(
        64, 7, strides=2, padding='same', name='conv1')(x)
    x = BatchNormalization(name='bn_conv1')(x)
    x = LeakyReLU(alpha=0.3)(x)
    print('COMPRESSION')
    print(c.audio_size, '-> ', x.get_shape().as_list())

    for i in range(1, c.n_compress_block+1):
        net[i] = x
        print(x.get_shape().as_list(), '-> ', end='')
        x = conv_block(x, 3,
                       [i**2*c.convo_size, i**2*c.convo_size, 4*i**2*c.convo_size],
                       stage=i, block='a')
        print(x.get_shape().as_list())    
        x = identity_block(x, 3,
                           [i**2*c.convo_size, i**2*c.convo_size, 4*i**2*c.convo_size],
                           stage=i, block='b')
        x = identity_block(x, 3,
                           [i**2*c.convo_size, i**2*c.convo_size, 4*i**2*c.convo_size],
                           stage=i, block='c')
    [print(k, v) for k, v in net.items()]
    print('\nAfter compression', x, '\n')
    
    # DECOMPRESS
    print('DECOMPRESSION')
    for i in range(c.n_compress_block, 0, -1):
        print(i, end=' ')
        print(x.get_shape().as_list(), '-> ', end='')
        x = deconv_block(x, 3,
                       [i**2*c.convo_size, i**2*c.convo_size, 4*i**2*c.convo_size],
                       stage=i, block='a_incr')
        print(x.get_shape().as_list())
        x = Concatenate(axis=2)([net[i], x])
        x = identity_block(x, 3,
                           [i**2*c.convo_size, i**2*c.convo_size, 4*i**2*c.convo_size],
                           stage=i, block='b_incr')
        x = identity_block(x, 3,
                           [i**2*c.convo_size, i**2*c.convo_size, 4*i**2*c.convo_size],
                           stage=i, block='c_incr')
    print(x.get_shape().as_list(), '-> ', end='')
    x = deconv_block(x, 3,
                     [i**2*c.convo_size, i**2*c.convo_size, 4*i**2*c.convo_size],
                     stage=42, block='a_incr')
    x = identity_block(x, 3,
                       [i**2*c.convo_size, i**2*c.convo_size, 4*i**2*c.convo_size],
                       stage=42, block='c_incr')
    print(x.get_shape().as_list())
    x = Conv1D(1, 1, strides=1, padding='same')(x)
    x = Reshape((-1,))(x)
    signal_output = Activation('tanh')(x)
    print('Recovered tensor', signal_output)
    # Create model.
    model = Model(signal_input, signal_output)

    return model

