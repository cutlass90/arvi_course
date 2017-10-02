import os
import argparse
import fnmatch

import numpy as np
import scipy
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
from keras.layers import Flatten

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
    if save_path:
        np.save(save_path, embeddings)
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

def get_model():
    base_resnet = ResNet50(include_top=False,
                           weights='imagenet',
                           input_shape=(224, 224, 3),
                           classes=1000)
    flatFeaturesLayer = Flatten()(base_resnet.output)
    featureExtractor = Model(inputs=base_resnet.input, outputs=flatFeaturesLayer)
    featureExtractor.compile('sgd', 'mse')
    return featureExtractor

def cos_dist(matrix, vector):
    """ Return cosine distance as 1D array. """
    v = vector.reshape(1, -1)
    return scipy.spatial.distance.cdist(matrix, v, 'cosine').reshape(-1)

def main(path, topn):
    model = get_model()
    paths = find_files(PATH_TO_DATA, '*.jpg')[:n_images]
    embeddings = make_embeddings(paths, model, batch_size)

    img = preprocess_img(path)
    img_emb = model.predict(img[None, ...])[0]

    img_distances = cos_dist(embeddings, img_emb)
    nearest_inds = np.argsort(img_distances)[:topn].tolist()
    top_path = np.array(paths)[nearest_inds]
    print(top_path)
    return top_path



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str,
                        help="path to desired image")
    parser.add_argument('topn', type=int,
                        help='number of images to search')

    args = parser.parse_args()
    main(args.path, args.topn)