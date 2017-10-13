import os
import fnmatch
import math
from operator import itemgetter

import numpy as np
from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.preprocessing.image import (ImageDataGenerator, Iterator,
                                       array_to_img, img_to_array, load_img)


def scaled_exp_decay(start: float, end: float, n_iter: int,
               current_iter: int) -> float:
    """ Exponentially modifies value from start to end.

    Args:
        start: float, initial value
        end: flot, final value
        n_iter: int, total number of iterations
        current_iter: int, current iteration
    """
    b = math.log(start/end, n_iter)
    a = start*math.pow(1, b)
    value = a/math.pow((current_iter+1), b)
    return value

def linear_decay(start: float, end: float, n_iter: int,
           current_iter: int) -> float:
    """ Linear modifies value from start to end.

    Args:
        start: float, initial value
        end: flot, final value
        n_iter: int, total number of iterations
        current_iter: int, current iteration
    """
    return (end - start)/n_iter*current_iter + start



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


def lr_scheduler(current_epoch, config):
    lr_min = config.train.lr_min
    lr_max = config.train.lr_max
    edge = config.train.epochs // 3
    if current_epoch < current_epoch:
        return linear_decay(lr_min, lr_max, edge, current_epoch)
    else:
        return linear_decay(lr_max, lr_min, config.train.epochs, current_epoch)

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

def read_dirs_names(path):
    """ Return list with dirs names for provided path. """
    return [name for name in os.listdir(path) if os.path.isdir(name)]

def read_data(img_paths, masks_paths, config):
    train_x = []
    train_masks = []
    for img, mask in zip(img_paths, masks_paths):
        train_x.append(image.img_to_array(image.load_img(img,
            target_size=(config.data.img_height, config.data.img_width))))
        train_masks.append(np.load(mask))
    train_x = np.stack(train_x)
    train_masks = np.stack(train_masks)
    return train_x, train_masks

def get_generators(config):
    def except_catcher(gen):
        while True:
            try:
                data = next(gen)
                yield data
            except Exception as e:
                print('Ups! Something wrong!', e)
    
    def normilize(img):
        return (img/255 - 0.5)*2

    train_datagen = ImageDataGenerator(preprocessing_function=normilize)

    test_datagen = ImageDataGenerator(preprocessing_function=normilize)

    img_paths = sorted(find_files(config.data.path_to_data, '*.jpg'))
    print(len(img_paths))
    masks_paths = sorted(find_files(config.data.path_to_masks, '*.npy'))
    print(len(masks_paths))
    train_img_paths,\
    test_img_paths,\
    train_masks_paths,\
    test_masks_paths = train_test_split(img_paths, masks_paths,
                                        test_size=config.data.test_size)

    train_x, train_masks = read_data(train_img_paths, train_masks_paths, config)
    test_x, test_masks = read_data(test_img_paths, test_masks_paths, config)

    train_generator = train_datagen.flow(
        x=train_x,
        y=train_masks,
        batch_size=config.data.batch_size,
        save_to_dir='./save_to_dir_train')

    validation_generator = test_datagen.flow(
        x=test_x,
        y=test_masks,
        batch_size=config.data.batch_size,
        save_to_dir='./save_to_dir_test')

    return train_generator, validation_generator
