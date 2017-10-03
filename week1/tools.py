import os
import fnmatch
from operator import itemgetter
import math

from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input

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


def lr_scheduler(current_epoch, epochs):
    edge = epochs // 3
    if current_epoch < current_epoch:
        return linear_decay(1e-5, 1e-3, edge, current_epoch)
    else:
        return linear_decay(1e-3, 1e-5, epochs, current_epoch)

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

def read_dirs_names(path):
    """ Return list with dirs names for provided path. """
    return [name for name in os.listdir(path) if os.path.isdir(name)]
############################# DATA PROCESSING ##################################
def preprocess_img(img):
    """ Load and preprocess one image. """
    img = image.img_to_array(image.load_img(img, target_size=(224, 224)))
    img = preprocess_input(img)
    return img
################################################################################