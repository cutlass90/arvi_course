import os

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.callbacks import EarlyStopping, ProgbarLogger, ModelCheckpoint
from keras.callbacks import LearningRateScheduler, CSVLogger, TensorBoard
from keras.layers import Dense, GlobalAveragePooling2D, Input, Flatten
from keras.optimizers import SGD
from keras.models import load_model
import numpy as np

import tools
from config import config


def get_model(num_classes, n_fozen_layers, path_to_weights_load):
    # prevent allocation all memory
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    set_session(tf.Session(config=tf_config))

    model = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    x = model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(num_classes*2, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=model.input, outputs=predictions)

    for layer in model.layers[:-n_fozen_layers]:
        layer.trainable = False

    model.compile(
        optimizer=SGD(lr=0.0001, momentum=0.9),
        loss='categorical_crossentropy',
        metrics=['accuracy'])
           
    # if exist load weights
    if path_to_weights_load:
        model.load_weights(path_to_weights_load)
        print('WEIGHTS {} WERE LOADED'.format(path_to_weights_load))
    else:
        print('IMAGENET WEIGHTS WERE LOADED')
    # model.summary()
    return model

def make_train_episode(model, layers, num_img, train_gen, valid_gen):
    path_to_weights = os.path.join(config.train.path_to_models,
                                   'model{}'.format(layers))
    call_backs =[
        EarlyStopping('val_acc', min_delta=1e-5, patience=20),
        ProgbarLogger('steps'),
        ModelCheckpoint(path_to_weights, save_best_only=True, save_weights_only=True),
        LearningRateScheduler(lambda x: tools.lr_scheduler(x, config)),
        CSVLogger(config.train.path_to_log),
        TensorBoard(config.train.path_to_summaries)
        ]

    steps_per_epoch=int((1-config.data.valid_size)*num_img/config.data.batch_size) // 1
    validation_steps=int(config.data.valid_size*num_img/config.data.batch_size) // 1
    model.fit_generator(generator=train_gen,
                        steps_per_epoch=steps_per_epoch,
                        epochs=config.train.epochs,
                        verbose=1,
                        callbacks=call_backs,
                        validation_data=valid_gen,
                        validation_steps=validation_steps,
                        max_queue_size=config.train.max_queue_size,
                        workers=config.train.workers,
                        use_multiprocessing=False)

def main():
    os.makedirs(config.train.path_to_summaries, exist_ok=True)
    os.makedirs(config.train.path_to_models, exist_ok=True)

    num_classes = len(os.listdir(config.data.path_to_train))
    print("Number of classes found: {}".format(num_classes))
    
    # num_img = len(tools.find_files(config.data.path_to_data, '*.jpg'))
    num_img = 652782
    print("Number of images found: {}".format(num_img))

    # image_lists = tools.create_image_lists(config.data.path_to_data,
    #                                        config.data.valid_size*100)

    train_gen, valid_gen = tools.get_generators_standart(config)

    for i, layers in enumerate(config.train.n_fozen_layers):
        if i == 0:
            path_to_weights_load = None
        else:
            path_to_weights_load = os.path.join(config.train.path_to_models,
                'model{}'.format(config.train.n_fozen_layers[i-1]))
        
        #path_to_weights_load = os.path.join(config.train.path_to_models,
        #        'model{}'.format('20'))
        #layers = 20/

        model = get_model(num_classes, layers, path_to_weights_load)
        make_train_episode(model, layers, num_img, train_gen, valid_gen)
    model.save_weights('.models/final_model')

if __name__ == '__main__':
    main()