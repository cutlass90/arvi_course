import os

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.callbacks import EarlyStopping, ProgbarLogger, ModelCheckpoint
from keras.callbacks import LearningRateScheduler, CSVLogger, TensorBoard
from keras.layers import Dense, GlobalAveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.models import load_model
import numpy as np

from config import config
import tools
from vggUnet import VGGUnet

train_gen, test_gen = tools.get_generators(config)
model = VGGUnet(config)

model.compile(
    optimizer=Adam(),
    loss='binary_crossentropy',
    metrics=['accuracy'])

os.makedirs(config.train.path_to_summaries, exist_ok=True)
os.makedirs(config.train.path_to_models, exist_ok=True)

call_backs =[
    EarlyStopping('val_acc', min_delta=1e-5, patience=20),
    ProgbarLogger('steps'),
    ModelCheckpoint(config.train.path_to_models+'/model', save_best_only=True),
    LearningRateScheduler(lambda x: tools.lr_scheduler(x, config)),
    CSVLogger(config.train.path_to_log),
    TensorBoard(config.train.path_to_summaries)
    ]

steps_per_epoch=int((1-config.data.test_size)*1700/config.data.batch_size) // 1
validation_steps=int(config.data.test_size*1700/config.data.batch_size) // 1
model.fit_generator(generator=train_gen,
                    steps_per_epoch=steps_per_epoch,
                    epochs=config.train.epochs,
                    verbose=1,
                    callbacks=call_backs,
                    validation_data=test_gen,
                    validation_steps=validation_steps,
                    max_queue_size=config.train.max_queue_size,
                    workers=config.train.workers,
                    use_multiprocessing=False)