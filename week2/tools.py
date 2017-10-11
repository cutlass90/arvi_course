import os
import fnmatch
import math
import random
from operator import itemgetter

import numpy as np
from pycocotools.coco import COCO
from keras.preprocessing import image
from scipy.misc import imresize



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

def normilize(img):
    return (img/255 - 0.5)*2


def generator(coco, c, path_to_imgs):
    img_ids = coco.getImgIds()
    cat_to_class_map = {cat: i for i, cat in enumerate(coco.getCatIds())}
    while True:
        random.shuffle(img_ids)
        X = get_imgs_by_ids(img_ids[:c.batch_size], coco, c, path_to_imgs)
        Y = get_targets_by_ids(img_ids[:c.batch_size], coco, c, cat_to_class_map)
        yield (X, Y)
    

def get_imgs_by_ids(img_ids, coco, c, path_to_imgs):
    """ Get list of img ids and return batch of images. """
    imgs = coco.loadImgs(img_ids)
    X = np.empty((c.batch_size, c.img_height, c.img_width, 3), dtype=np.float)
    for i, img in enumerate(imgs):
        path = os.path.join(path_to_imgs, img['file_name'])
        X[i] = image.img_to_array(image.load_img(path,
            target_size=[c.img_height, c.img_width]))/127.5 - 1
    return X

def get_targets_by_ids(img_ids, coco, c, cat_to_class_map):
    """ Return targets as array batch_size x height x width x n_channels. """
    Y = np.zeros((c.batch_size, c.img_height, c.img_width, c.n_classes),
                    dtype=np.float)
    for b, id_ in enumerate(img_ids):
        anns = coco.imgToAnns[id_]
        for ann in anns:
            i = cat_to_class_map[ann['category_id']]
            mask = imresize(coco.annToMask(ann), [c.img_height, c.img_width])
            mask = (mask > 0).astype(float)
            Y[b, :, :, i] += mask
    Y = (Y > 0).astype(np.uint8)
    return Y

def get_generators(c):
    train_coco = COCO(c.path_to_train_json)
    test_coco = COCO(c.path_to_test_json)
    
    train_gen = generator(train_coco, c, c.path_to_train_imgs)
    test_gen = generator(test_coco, c, c.path_to_test_imgs)
    return except_catcher(train_gen), except_catcher(test_gen)

    

