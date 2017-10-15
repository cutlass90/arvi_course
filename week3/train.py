from keras.models import Sequential
from keras.layers import Dense, Activation, TimeDistributed
from keras.layers import LSTM
from keras.callbacks import EarlyStopping, ProgbarLogger, ModelCheckpoint
from keras.callbacks import LearningRateScheduler, CSVLogger, TensorBoard
from keras.optimizers import RMSprop
from keras.models import load_model
import numpy as np
import random
import sys
import os

import tools
from config import config as c


paths = tools.find_files(c.path_to_texts, '*.txt')
random.shuffle(paths)
# edge = int(len(paths)*c.test_size)
# train_paths = paths[:-edge]
# test_paths = paths[-edge:]

char_to_ind = tools.get_dict()
dict_size = len(char_to_ind)

train_gen = tools.get_generator(paths, char_to_ind, c.max_len, c.batch_size)
# train_gen = tools.get_generator(train_paths, char_to_ind, c.max_len, c.batch_size)
# test_gen = tools.get_generator(test_paths, char_to_ind, c.max_len, c.batch_size)

def get_model():
    if os.path.isfile(c.path_to_models+'/model_all'):
        model = load_model(c.path_to_models+'/model_all')
        print('model loaded')
    else:
        model = Sequential()
        model.add(LSTM(256, return_sequences=True, input_shape=(c.max_len, dict_size)))
        model.add(LSTM(256, return_sequences=True, input_shape=(c.max_len, 256)))
        model.add(LSTM(256, return_sequences=True, input_shape=(c.max_len, 256)))
        model.add(TimeDistributed(Dense(dict_size)))
        model.add(Activation('softmax'))
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy')
    return model

model = get_model()

os.makedirs(c.path_to_summaries, exist_ok=True)
os.makedirs(c.path_to_models, exist_ok=True)

call_backs =[
    # EarlyStopping('val_acc', min_delta=1e-5, patience=20),
    ProgbarLogger('steps'),
    ModelCheckpoint(c.path_to_models+'/model_OE', save_best_only=True),
    # LearningRateScheduler(lambda x: tools.lr_scheduler(x, c)),
    # CSVLogger(c.path_to_log),
    TensorBoard(c.path_to_summaries)
    ]

model.fit_generator(generator=train_gen,
                    steps_per_epoch=200,
                    epochs=c.epochs,
                    verbose=1,
                    callbacks=call_backs,
                    # validation_data=test_gen,
                    # validation_steps=50,
                    max_queue_size=c.max_queue_size,
                    workers=c.workers)
model.save(c.path_to_models+'/model_OE')




