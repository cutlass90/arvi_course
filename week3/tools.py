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
import numpy as np
from scipy.ndimage import imread
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

import landmarks as land



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

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

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

def fake_generator(c):
    while True:
        img_inputs = np.ones([c.batch_size, c.n_frames, c.img_height, c.img_width, 1])
        landmark_inputs = np.ones([c.batch_size, c.n_frames, c.landmark_size])
        out_emotion = np.zeros([c.batch_size, c.n_emotions])
        out_emotion[:, 0] = 1
        out_au = np.zeros([c.batch_size, c.n_action_units])
        out_au[:, 0] = 1
        yield ({'img_inputs': img_inputs, 'landmark_inputs': landmark_inputs},
            {'out_emotion': out_emotion, 'out_au': out_au})

def resample_imgs(imgs, target_frames):
    # imgs - 3D array frame height width
    # target_frames: int, target number of frames
    return nd.interpolation.zoom(imgs, zoom=[target_frames/len(imgs), 1, 1])


def process_single_img(img, c):
    if c.rectangle_imgs:
        shape = img.shape
        min_shape, max_shape = np.min(shape), np.max(shape)
        l = (max_shape - min_shape)//2
        r = (max_shape - min_shape) - l
        if shape[0] > shape[1]:
            img = img[l:-r, :]
        else:
            img = img[:, l:-r]
    if img.shape != (c.img_height, c.img_width):
        img = imresize(img, (c.img_height, c.img_width))
    return img

def data_augmentation(imgs, c):
    # imgs - list of 2D images

    def sequence_change(imgs, change_faktor):
        # change_faktor: float from 0 to 0.4
        l = random.randint(0, int(len(imgs)*change_faktor))
        r = random.randint(0, int(len(imgs)*change_faktor)) + 1
        return imgs[l:-r]
    imgs = sequence_change(imgs, c.sequence_change)
    imgs = np.stack(imgs)
    imgs = resample_imgs(imgs, c.n_frames)
    imgs = image.random_zoom(imgs, c.zoom_range)
    imgs = image.random_rotation(imgs, c.random_rotation)
    imgs = image.random_shear(imgs, c.random_shear)
    return imgs

def process_imgs(paths, c, augmentation=True):
    imgs = []
    for img in paths:
        img = imread(img, flatten=True, mode='F').astype(np.uint8)
        imgs.append(process_single_img(img, c))
    if augmentation:
        imgs = data_augmentation(imgs, c)
    else:
        imgs = np.stack(imgs)
        imgs = resample_imgs(imgs, c.n_frames)
    return imgs

def process_landmarks(imgs, path):
    landmarks = []
    for img in imgs:
        landmarks.append(np.reshape(land.gen_landmark(img, path)[1], [-1]))
    landmarks_delta = []
    for i in range(len(landmarks) - 1):
        landmarks_delta.append(landmarks[i] - landmarks[i+1])
    landmarks_delta.append(landmarks[-1] - landmarks[0])
    return np.hstack([np.stack(landmarks), np.stack(landmarks_delta)])

def generator(c, paths):
    while True:
        random.shuffle(paths)
        img_inputs = np.empty([c.batch_size, c.n_frames, c.img_height, c.img_width])
        landmark_inputs = np.zeros([c.batch_size, c.n_frames, c.landmark_size])
        out_emotion = np.zeros([c.batch_size, c.n_emotions])
        out_au = np.zeros([c.batch_size, c.n_action_units])
        for b, path in enumerate(paths[:c.batch_size]):
            # img_inputs
            img_paths = find_files(os.path.join(c.path_to_data, 'images/', path), '*.png')
            imgs = process_imgs(img_paths, c, augmentation=True)
            img_inputs[b] = imgs.astype(float)/127.5 - 1

            # landmark_inputs
            landmark_inputs[b] = process_landmarks(imgs, path)

            # out_emotion
            emo_path = find_files(os.path.join(c.path_to_data, 'emotions/', path), '*.txt')
            if len(emo_path) > 0:
                with open(emo_path[0], 'r') as f:
                    class_ = int(float(f.read()[3:-1]))
                out_emotion[b, class_-1] = 1

            #out_au
            label = find_files(os.path.join(c.path_to_data, 'labels/', path), '*.txt')[0]
            au = pd.read_csv(label, delimiter='   ', names=['emotion', 'value'])
            for e in au.emotion:
                out_au[b, c.au_map[e]] = 1
        
        img_inputs = np.expand_dims(img_inputs, 4)
        yield ({'img_inputs': img_inputs, 'landmark_inputs': landmark_inputs},
            {'out_emotion': out_emotion, 'out_au': out_au})



def get_generators(c):
    if os.path.isfile(c.saved_paths):
        with open(c.saved_paths, 'rb') as f:
            train_paths, test_paths = pickle.load(f)
        print('Paths were loaded')
    else:
        path = os.path.join(c.path_to_data, 'images/')
        paths = [os.path.join(g, l) for g in os.listdir(path) for l in os.listdir(path+g)
            if os.path.isdir(os.path.join(path, g, l))]
        edge = int(len(paths)*c.test_size)
        train_paths = paths[:-edge]
        test_paths = paths[-edge:]
        with open(c.saved_paths, 'wb') as f:
            pickle.dump((train_paths, test_paths), f)
        print('New paths were generated')
    train_gen = generator(c, train_paths)
    test_gen = generator(c, test_paths)
    return train_gen, test_gen


def masked_loss(y_true, y_pred):
    # when y_true == 0 mask loss
    loss = K.categorical_crossentropy(y_true, y_pred)
    mask = tf.cast(tf.greater(tf.reduce_sum(y_true, axis=1), 0.5), tf.float32)
    # mask = tf.Print(mask, [mask, loss], message='\n\n!!!mask, loss', summarize=20)
    return tf.reduce_mean(mask*loss)

def masked_acc(y_true, y_pred):
    # when y_true == 0 mask loss
    acc = K.cast(K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1)),
           K.floatx())
    mask = tf.cast(tf.greater(tf.reduce_sum(y_true, axis=1), 0.5), tf.float32)
    # mask = tf.Print(mask, [mask, acc], message='\n\n!!!mask, acc', summarize=20)
    return tf.reduce_sum(acc*mask)/(tf.reduce_sum(mask) + 1e-6)


#------------------------- Augmentation ---------------------------------------

################################################################################

    

if __name__ == '__main__':
    from config import config as c
    train_gen, test_gen = get_generators(c)
    a = next(train_gen)
