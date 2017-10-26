import os
import pickle
import fnmatch
import math
import random
from operator import itemgetter
from collections import defaultdict
from functools import partial

import tensorflow as tf
from keras import backend as K
from keras.layers import Conv2DTranspose, Lambda, Conv1D
from keras.legacy import interfaces
from keras.engine import InputSpec
from keras.utils import conv_utils


import numpy as np
from scipy.ndimage import imread
from keras.layers import Input, Reshape, Flatten
from keras import layers
from keras.models import Model
from keras.layers import Dense, Reshape
from keras.preprocessing import image
from keras_vggface.utils import preprocess_input
from scipy.misc import imresize
from keras import backend as K
import tensorflow as tf
from scipy.ndimage import imread
from scipy.misc import imsave, imresize, toimage
from scipy.signal import resample
from scipy import ndimage as nd
import pandas as pd
from PIL import Image




def _get_fields(attr):
    if isinstance(attr, Config):
        return [getattr(attr, k) for k in
                sorted(attr.__dict__.keys())]
    else:
        return [attr]


class Config:

    def __init__(self, **kwargs):
        """ Init config class with local configurations provided via kwargs.
            Provide scope field to mark config as main.
        """

        for name, attr in kwargs.items():
            setattr(self, name, attr)

        if 'scope' in kwargs.keys():
            self.is_main = True

            # collect all fields from all configs and regular kwargs
            fields = (_get_fields(attr) for name, attr in
                      sorted(kwargs.items(), key=itemgetter(0))
                      if not name == "scope")

            self.identifier_fields = sum(fields, [])

    @property
    def identifier(self):
        if self.is_main:
            fields = "_".join(self._process_attr(name)
                              for name in self.identifier_fields)
            return self.scope + "_" + fields
        else:
            raise AttributeError("There is no field `scope` in this config")

    def _process_attr(self, attr):
        if isinstance(attr, (int, float, str)):
            return str(attr)
        elif isinstance(attr, (list, tuple)):
            return 'x'.join(str(a) for a in attr)
        elif isinstance(attr, bool):
            if attr:
                return 'YES{}'.format(str(attr))
            else:
                return 'NO{}'.format(str(attr))
        else:
            raise TypeError('Wrong dtype.')


def find_files(path: str, filename_pattern: str, sort: bool = True) -> list:
    """Finds all files of type `filename_pattern`
    in directory and subdirectories paths.

    Args:
        path: str, directory to search files.
        filename_pattern: regular expression to specify file type.
        sort: bool, whether to sort files list. Defaults to True.

    Returns: list of found files.
    """
    files = list()
    for root, _, filenames in os.walk(path):
        for filename in fnmatch.filter(filenames, filename_pattern):
            files.append(os.path.join(root, filename))
    if sort:
        files.sort()
    return files

def get_file_name(filepath: str) -> str:
    """Returns file name without extension and directory

    Args:
        filepath: str, filepath.

    Returns:
        str, filename
    """

    f = os.path.basename(filepath)
    filename, _ = os.path.splitext(f)

    return filename

def except_catcher(gen):
    while True:
        try:
            i=0
            data = next(gen)
            yield data
        except Exception as e:
            i += 1
            print('Ups! Something wrong!', e)
            if i > 100:
                print('Can not yield data 100 times.')
                raise Exception('Sumething realy wrong!')

def Conv1DTranspose(inputs, filters, kernel_size, strides, padding):
    input_sh = inputs.get_shape().as_list()
    x = Reshape([input_sh[1], 1, input_sh[2]])(inputs)
    transposer = Conv2DTranspose(filters=filters,
                        kernel_size=(kernel_size, 1),
                        strides=(strides, 1),
                        padding=padding)
    x2 = transposer(x)
    out_sh = transposer.compute_output_shape([input_sh[0], input_sh[1], 1, input_sh[2]])
    out = Reshape([out_sh[1], out_sh[3]])(x2)
    return out

def categorical_crossentropy_gan(y_true, y_pred):
    """ Compute categorical_crossentropy for first half of batch:
        from 0 to batch/2"""
    loss = K.categorical_crossentropy(y_true, y_pred)
    half = tf.cast(tf.shape(y_true)[0]/2, tf.int32)   
    return tf.reduce_mean(loss[:half])


def fake_generator(c):
    DICT_SIZE = len(c.char_to_class)
    while True:
        real_signal = np.random.random_sample(size=[c.batch_size, c.audio_size])*2 - 1
        fake_signal = np.random.random_sample(size=[c.batch_size, c.audio_size])*2 - 1
        real_text = np.random.randint(0, 1, size=[c.batch_size, c.text_size, DICT_SIZE])
        yield real_signal, fake_signal, real_text
# class Conv1DTranspose(Conv2DTranspose):
#     def __init__(self, filters,
#                  kernel_size,
#                  strides=1,
#                  padding='valid',
#                  data_format=None,
#                  activation=None,
#                  use_bias=True,
#                  kernel_initializer='glorot_uniform',
#                  bias_initializer='zeros',
#                  kernel_regularizer=None,
#                  bias_regularizer=None,
#                  activity_regularizer=None,
#                  kernel_constraint=None,
#                  bias_constraint=None,
#                  **kwargs):
#         super().__init__(
#             filters,
#             kernel_size=(kernel_size, 1),
#             strides=(strides, 1),
#             padding=padding,
#             data_format=data_format,
#             activation=activation,
#             use_bias=use_bias,
#             kernel_initializer=kernel_initializer,
#             bias_initializer=bias_initializer,
#             kernel_regularizer=kernel_regularizer,
#             bias_regularizer=bias_regularizer,
#             activity_regularizer=activity_regularizer,
#             kernel_constraint=kernel_constraint,
#             bias_constraint=bias_constraint,
#             **kwargs)
#         self.input_spec = InputSpec(ndim=3)
    
#     def call(self, inputs):
#         x = K.expand_dims(inputs, axis=2)
#         x = super().call(x)
#         sh = list(self.compute_output_shape(inputs.get_shape().as_list()))
#         sh[0] = -1
#         out = K.reshape(x, sh)
#         return out

#     def build(self, input_shape):
#         if len(input_shape) != 3:
#             raise ValueError('Inputs should have rank ' +
#                              str(3) +
#                              '; Received input shape:', str(input_shape))
#         if self.data_format == 'channels_first':
#             channel_axis = 1
#         else:
#             channel_axis = -1
#         if input_shape[channel_axis] is None:
#             raise ValueError('The channel dimension of the inputs '
#                              'should be defined. Found `None`.')
#         input_dim = input_shape[channel_axis]
#         kernel_shape = self.kernel_size + (self.filters, input_dim)

#         self.kernel = self.add_weight(shape=kernel_shape,
#                                       initializer=self.kernel_initializer,
#                                       name='kernel',
#                                       regularizer=self.kernel_regularizer,
#                                       constraint=self.kernel_constraint)
#         if self.use_bias:
#             self.bias = self.add_weight(shape=(self.filters,),
#                                         initializer=self.bias_initializer,
#                                         name='bias',
#                                         regularizer=self.bias_regularizer,
#                                         constraint=self.bias_constraint)
#         else:
#             self.bias = None
#         # Set input spec.
#         self.input_spec = InputSpec(ndim=3, axes={channel_axis: input_dim})
#         self.built = True
    
#     def compute_output_shape(self, input_shape):
#         output_shape = list(input_shape)
#         output_shape = output_shape[:2] + [1] + output_shape[-1:]
#         if self.data_format == 'channels_first':
#             c_axis, h_axis, w_axis = 1, 2, 3
#         else:
#             c_axis, h_axis, w_axis = 3, 1, 2

#         kernel_h, kernel_w = self.kernel_size
#         stride_h, stride_w = self.strides

#         output_shape[c_axis] = self.filters
#         output_shape[h_axis] = conv_utils.deconv_length(
#             output_shape[h_axis], stride_h, kernel_h, self.padding)
#         output_shape[w_axis] = conv_utils.deconv_length(
#             output_shape[w_axis], stride_w, kernel_w, self.padding)
#         output_shape = tuple(output_shape)
#         output_shape = (output_shape[0], output_shape[1], output_shape[3])
#         return output_shape
    
################################################################################

    

if __name__ == '__main__':
    inp = Input(shape=(100,))
    x = Dense(200)(inp)
    x = Reshape([100, 2])(x)
    # out = Conv1DTranspose(10, 3, 2, padding='same')(x)
    x = Conv1DTranspose(x, 10, 3, 2, 'same')
    print(x)
    x = Conv1D(1, 1, strides=1, padding='same')(x)
    x = Flatten()(x)
    out = Dense(20)(x)
    model = Model(inp, out)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam')
    model.fit(x=np.ones([100, 100]),
              y=np.ones([100, 20]))