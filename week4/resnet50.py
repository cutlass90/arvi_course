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
from keras.layers import BatchNormalization, Reshape
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
        print('convert ', input_tensor.get_shape().as_list()[-1], 'to', filters3)
        input_tensor = Conv1D(filters3, 1, name=conv_name_base + '2a')(input_tensor)
        input_tensor = BatchNormalization(name=bn_name_base + '2a')(input_tensor)
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

    x = Conv1DTranspose(filters1, 1, strides=strides,
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = LeakyReLU(alpha=0.3)(x)

    x = Conv1DTranspose(filters2, kernel_size, padding='same',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = LeakyReLU(alpha=0.3)(x)

    x = Conv1DTranspose(filters3, 1, name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    shortcut = Conv1DTranspose(filters3, 1, strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = LeakyReLU(alpha=0.3)(x)
    return x

def ResNetDiscriminator(c):
    """
    Returns
        A Keras model instance.
    """
    signal_input = Input(shape=(c.audio_size, 1))
    x = Conv1D(
        64, 7, strides=2, padding='same', name='conv1')(signal_input)
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
    # Create model.
    model = Model(signal_input, x, name='resnet50')

    return model


def ResNetGenerator(c):
    """
    Returns
        A Keras model instance.
    """
    net = {}
    signal_input = Input(shape=(c.audio_size, 1))

    # COMPRESS
    x = Conv1D(
        64, 7, strides=2, padding='same', name='conv1')(signal_input)
    x = BatchNormalization(name='bn_conv1')(x)
    x = LeakyReLU(alpha=0.3)(x)
    print(c.audio_size, '->', x.get_shape().as_list())

    for i in range(1, c.n_compress_block+1):
        net[i] = x
        print(x.get_shape().as_list(), '->', end='')
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
    print('\nAfter compression',x)
    
    # DECOMPRESS
    for i in range(c.n_compress_block, 0, -1):
        print(i, end=' ')
        print(x.get_shape().as_list(), '->', end='')
        x = deconv_block(x, 3,
                       [i**2*c.convo_size, i**2*c.convo_size, 4*i**2*c.convo_size],
                       stage=i, block='a')
        print(x.get_shape().as_list())    
        x = tf.concat([net[i], x], axis=2)
        x = identity_block(x, 3,
                           [i**2*c.convo_size, i**2*c.convo_size, 4*i**2*c.convo_size],
                           stage=i, block='b')
        x = identity_block(x, 3,
                           [i**2*c.convo_size, i**2*c.convo_size, 4*i**2*c.convo_size],
                           stage=i, block='c')
    x = Conv1D(1, 1, strides=1, padding='same')(x)
    print(x)
    x = Reshape((-1,))(x)
    print(x)
    # Create model.
    model = Model(signal_input, x, name='resnet50')

    return model

