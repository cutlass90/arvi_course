# -*- coding: utf-8 -*-
import os
import fnmatch
import math
import random
from operator import itemgetter
from collections import defaultdict
import re

import numpy as np

# CHARS = '\n !&(),-./0123456789:;?ABHIKMOVX_`aceijmopqrtxy| «»ЄІЇАБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЬЮЯабвгдежзийклмнопрстуфхцчшщыьэюяєіїҐґ—’…'
CHARS = '\n !&(),-./0123456789:;?ABHIKMOVX_`aceijmopqrtxy|«»ЄІЇАБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЬЮЯабвгдежзийклмнопрстуфхцчшщыьэюяєіїҐґ—’…'


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


def get_dict():
    char_to_ind = {ch:i for i, ch in enumerate(CHARS)}
    return char_to_ind

def get_generator(paths, char_to_ind, max_len, batch_size):
    # char_to_ind - dict
    while True:
        file_ = random.sample(paths, 1)[0]
        with open(file_, 'r') as f:
            text = f.read()
        # remove all char not in dict
        text = re.sub('[^'+CHARS+']', '', text)

        data = np.zeros(shape=[batch_size, max_len+1, len(char_to_ind)],
                    dtype=np.int8)
        for b in range(batch_size):
            s = random.randint(0, len(text)-max_len-1)
            for i, ch in enumerate(text[s:s+max_len+1]):
                data[b, i, char_to_ind[ch]] = 1
        yield data[:, :-1, :], data[:, 1:, :]
    
def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def generate_text(model, init_sentence, diversity, size, char_to_ind):
    ind_to_char = {v:k for k, v in char_to_ind.items()}
    init_sentence = re.sub('[^'+CHARS+']', '', init_sentence)
    generated = ''
    for i in range(size):
        x = np.zeros((1, len(init_sentence), len(char_to_ind)))
        for t, char in enumerate(init_sentence):
            x[0, t, char_to_ind[char]] = 1.
        preds = model.predict(x, verbose=0)[0][-1, :]
        next_index = sample(preds, diversity)
        next_char = ind_to_char[next_index]
        generated += next_char
        init_sentence = init_sentence[1:] + next_char
    return generated