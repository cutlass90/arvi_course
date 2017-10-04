import os

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.callbacks import EarlyStopping, ProgbarLogger, ModelCheckpoint
from keras.callbacks import LearningRateScheduler, CSVLogger, TensorBoard
from keras.layers import Dense, GlobalAveragePooling2D, Input, Flatten
from keras.optimizers import SGD
import numpy as np

import tools
from config import config

def get_model(num_classes):
    # prevent allocation all memory
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    set_session(tf.Session(config=tf_config))

    model = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    x = model.output
    x = GlobalAveragePooling2D()(x)
    # x = Flatten()(x)
    x = Dense(num_classes*2, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=model.input, outputs=predictions)

    for layer in model.layers[:-12]:
        layer.trainable = False

    model.compile(
        optimizer=SGD(lr=0.0001, momentum=0.9),
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    model.summary()
    return model

def main():
    os.makedirs('./summaries', exist_ok=True)
    os.makedirs('./models', exist_ok=True)

    num_classes = len(os.listdir(config.data.path_to_data))
    print("Number of classes found: {}".format(num_classes))
    
    # num_img = len(tools.find_files(config.data.path_to_data, '*.jpg'))
    num_img = 652782
    print("Number of images found: {}".format(num_img))

    image_lists = tools.create_image_lists(config.data.path_to_data,
                                           config.data.valid_size*100)

    train_gen, valid_gen = tools.get_generators(image_lists, config)
    # train_gen, valid_gen = tools.mock_generator(config), tools.mock_generator(config)

    model = get_model(num_classes)

    call_backs =[
        EarlyStopping('categorical_crossentropy', min_delta=1e-5, patience=2),
        ProgbarLogger('steps'),
        ModelCheckpoint('./models/model', save_best_only=True),
        LearningRateScheduler(lambda x: tools.lr_scheduler(x, config.train.epochs)),
        CSVLogger('./log.csv'),
        TensorBoard('./summaries')
    ]

    steps_per_epoch=int((1-config.data.valid_size)*num_img/config.data.batch_size) // 10
    validation_steps=int(config.data.valid_size*num_img/config.data.batch_size) // 10
    model.fit_generator(generator=train_gen,
                        steps_per_epoch=steps_per_epoch,
                        epochs=config.train.epochs,
                        verbose=1,
                        callbacks=call_backs,
                        validation_data=valid_gen,
                        validation_steps=validation_steps,
                        max_queue_size=config.train.max_queue_size,
                        workers=config.train.workers,
                        use_multiprocessing=True)


if __name__ == '__main__':
    main()