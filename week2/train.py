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

from config import config as c
import tools
from vggUnet import VGGUnet

train_gen, val_gen = tools.get_generators(c)
if os.path.isfile(c.path_to_models+'/model'):
    model = load_model(c.path_to_models+'/model')
    print('model is loaded')
else:
    model = VGGUnet(c)

model.compile(
    optimizer=Adam(),
    loss='binary_crossentropy',
    metrics=['accuracy'])

os.makedirs(c.path_to_summaries, exist_ok=True)
os.makedirs(c.path_to_models, exist_ok=True)

call_backs =[
    EarlyStopping('val_acc', min_delta=1e-5, patience=20),
    ProgbarLogger('steps'),
    ModelCheckpoint(c.path_to_models+'/model', save_best_only=True),
    # LearningRateScheduler(lambda x: tools.lr_scheduler(x, c)),
    CSVLogger(c.path_to_log),
    TensorBoard(c.path_to_summaries)
    ]

model.fit_generator(generator=train_gen,
                    steps_per_epoch=500,
                    epochs=c.epochs,
                    verbose=1,
                    callbacks=call_backs,
                    validation_data=val_gen,
                    validation_steps=100,
                    max_queue_size=c.max_queue_size,
                    workers=c.workers,
                    use_multiprocessing=False)