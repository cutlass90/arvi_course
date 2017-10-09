import os
import argparse
import fnmatch
from shutil import copy
import pdb

import pandas as pd
import numpy as np
import scipy
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
from keras.layers import Flatten
from keras.models import load_model
import scipy.misc

import tools
from config import config


def get_mask(args, path_to_img):
    model = load_model(args.path_to_model)
    img = image.img_to_array(image.load_img(path_to_img,
        target_size=[config.data.img_height, config.data.img_width]))
    img = (img/255 - 0.5)*2
    predict = model.predict(img[None, ...])[0]
    print(predict.shape)
    predict = predict[:, :, None] * np.ones(3, dtype=float)[None, None, :]
    print(predict.shape)
    predict = predict < 0.5


    img = image.img_to_array(image.load_img(path_to_img))
    original_size = img.shape[:2]

    bali = image.img_to_array(image.load_img('./bali.jpg', target_size=config.data.image_shape[:2]))

    img = scipy.misc.imresize(img, size=config.data.image_shape[:2])
    img[predict] = bali[predict]
    img = scipy.misc.imresize(img, size=original_size)
    path_to_save = os.path.join(args.path_to_results,
                                tools.get_file_name(path_to_img)+'.jpg')
    print(path_to_save)
    scipy.misc.imsave(path_to_save, img)

def main(args):
    os.makedirs(args.path_to_results, exist_ok=True)
    if os.path.isfile(args.path_to_img):
        get_mask(args, args.path_to_img)
    elif os.path.isdir(args.path_to_img):
        for img in tools.find_files(args.path_to_img, '*.jpg'):
            get_mask(args, img)
    else:
        raise AttributeError('You should provide correct path to img')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-path_to_img', type=str,
                        help="path to single image or path to dir")
    parser.add_argument('-path_to_model', type=str, default='./models/model',
                        help="path to saved model")
    parser.add_argument('-path_to_results', type=str, default='./results',
                        help="path where results will be saved")
    args = parser.parse_args()
    main(args)