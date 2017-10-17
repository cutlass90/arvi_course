import os
import fnmatch
import math
import random
from operator import itemgetter
from collections import defaultdict
from functools import partial

import numpy as np
from pycocotools.coco import COCO
from keras.preprocessing import image
from keras_vggface.utils import preprocess_input
from scipy.misc import imresize
from keras import backend as K
import tensorflow as tf
from scipy.ndimage import imread
from scipy.misc import imsave
from scipy.signal import resample
from scipy import ndimage as nd



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

def fake_generator(c):
    while True:
        img_inputs = np.ones([c.batch_size, c.n_frames, c.img_height, c.img_width, 3])
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

def generator(c, paths):
    while True:
        random.shuffle(paths)
        img_inputs = np.empty(c.batch_size, c.n_frames, c.img_height, c.img_width)
        landmark_inputs = np.ones([c.batch_size, c.n_frames, c.landmark_size])
        out_emotion = np.zeros([c.batch_size, c.n_emotions])
        out_au = np.zeros([c.batch_size, c.n_action_units])
        for b, path in enumerate(paths[:c.batch_size]):
            # img_inputs
            for img in find_files(os.path.join(c.path_to_data, 'images', path),
                                  '*.png'):
                imgs = []
                img = image.img_to_array(image.load_img(img,
                    target_size=(c.img_height, c.img_width)))
                img = preprocess_input(img)
                imgs.append(img)
            imgs = np.stack(imgs)
            imgs = resample_imgs(imgs, c.n_frames)
            img_inputs[b] = imgs

            # landmark_inputs
            for l in find_files(os.path.join(c.path_to_data, 'landmarks', path),
                                '*.txt'):
                pass
            landmark_inputs[b] = np.ones([c.n_frames, c.landmark_size])

            # out_emotion
            emo_path = find_files(os.path.join(c.path_to_data, 'emotions', path),
                                  '*.txt')
            if len(emo_path) == 0:
                class_ = random.randint(1, c.n_emotions)
            else:
                with open(emo_path[0], 'r') as f:
                    class_ = int(f.read())
            

            


        
        yield ({'img_inputs': img_inputs, 'landmark_inputs': landmark_inputs},
            {'out_emotion': out_emotion, 'out_au': out_au})



def get_generators(c):
    path = os.path.join(c.path_to_data, 'images')
    paths = [os.path.join(g, l) for g in os.listdir(path) for l in os.listdir(path+g)
        if os.path.isdir(os.path.join(path, g, l))]
    edge = int(len(paths)*c.test_size)
    train_paths = paths[:-edge]
    test_paths = paths[-edge:]
    train_gen = generator(c, train_paths)
    test_gen = generator(c, test_paths)
