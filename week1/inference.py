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

import tools
import train

############################# PARAMETERS #######################################
PATH_TO_DATA = '../datasets/coco/train2017/'
n_images = 1000 # number of images to search
PATH_TO_VECTORS = 'data/vectors_resnet/'
batch_size = 64
################################################################################

def make_embeddings(paths, model, batch_size, save_path=None):
    """ Compute vector representation of provided images.

    Args:
        paths: list of string with paths to images
        model: an instance of pretrained model
        batch_size: ind, batch size
        save: str, path to save vectors
    
    Return:
        embeddings: np.array, embedding vectors
    """
    embeddings = []
    for i in range(0, len(paths), batch_size):
        try:
            batch = paths[i:i+batch_size]
        except IndexError:
            batch = paths[i:len(paths)]
        batch = np.array([preprocess_img(img) for img in batch])
        embeddings.append(model.predict_on_batch(batch).astype(np.float16))
    embeddings = np.vstack(embeddings)
    embeddings = [emb for emb in embeddings]
    embeddings = pd.DataFrame({'path':paths, 'embedding':embeddings})
    print(embeddings.head())
    if save_path:
        embeddings.to_pickle(save_path)
    return embeddings

def preprocess_img(img):
    """ Load and preprocess one image. """
    img = image.img_to_array(image.load_img(img, target_size=(224, 224)))
    img = preprocess_input(img)
    return img

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



def cos_dist(matrix, vector):
    """ Return cosine distance as 1D array. """
    v = vector.reshape(1, -1)
    return scipy.spatial.distance.cdist(matrix, v, 'cosine').reshape(-1)

def get_prediction(args):
    classes_map = tools.get_classes_map('./autoria/train')
    model = train.get_model(514, 0, args.path_to_weights)
    img = image.img_to_array(image.load_img(args.path_to_img,
                                            target_size=(224, 224)))
    img = (img/255 - 0.5)*2
    predict = model.predict(img[None, ...])[0]
    predict = np.argmax(np.array(predict))
    to_write = 'predict class {} - {}\n'.format(predict, classes_map[predict])
    print(to_write)
    with open(os.path.join(args.path_to_results, 'results.txt'), 'w') as f:
        f.write(to_write)

def find_similar_images(args):
    model = train.get_model(514, 0, args.path_to_weights)
    flattet_layer = model.get_layer(index=-3).output
    feature_extractor = Model(inputs=model.inputs, outputs=flattet_layer)
    feature_extractor.compile('sgd', 'mse')

    if args.path_to_embeddings is None:
        paths = find_files('./autoria', '*.jpg')
        embeddings = make_embeddings(paths, feature_extractor, batch_size=64,
                                     save_path='./embeddings.pkl')
    else:
        embeddings = pd.read_pickle(args.path_to_embeddings)
    
    
    img = image.img_to_array(image.load_img(args.path_to_img,
                                            target_size=(224, 224)))
    img = (img/255 - 0.5)*2
    img_emb = feature_extractor.predict(img[None, ...])[0]

    img_distances = cos_dist(np.stack(embeddings.embedding), img_emb)
    nearest_inds = np.argsort(img_distances)[-args.topn:].tolist()
    top_path = embeddings.path[nearest_inds]
    with open(os.path.join(args.path_to_results, 'results.txt'), 'a') as f:
        f.write('Top {} similar images are:\n'.format(args.topn))
        for p in top_path:
            f.write(p+'\n')
    
    # copy to result path
    for path in top_path:
        copy(path, args.path_to_results)


def main(args):
    os.makedirs(args.path_to_results, exist_ok=True)
    get_prediction(args)
    find_similar_images(args)
    

    



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-path_to_img', type=str,
                        help="path to desired image")
    parser.add_argument('-path_to_weights', type=str,
                        help="path to saved model weights")
    parser.add_argument('-path_to_embeddings', type=str,
                        help="path to vector representation of images")
    parser.add_argument('-path_to_results', type=str, default='./results',
                        help="path where results will be saved")
    parser.add_argument('-topn', type=int, default=5,
                        help='number of images to search')
    args = parser.parse_args()
    main(args)