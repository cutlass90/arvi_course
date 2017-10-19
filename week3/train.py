import os

from keras.callbacks import EarlyStopping, ProgbarLogger, ModelCheckpoint
from keras.callbacks import LearningRateScheduler, CSVLogger, TensorBoard
from keras.optimizers import RMSprop
from keras.models import load_model

import tools
from facial_recognizer import facial_recognizer
from config import config as c

os.makedirs(c.path_to_summaries, exist_ok=True)
os.makedirs(c.path_to_models, exist_ok=True)

call_backs =[
    # EarlyStopping('val_acc', min_delta=1e-5, patience=20),
    ProgbarLogger('steps'),
    ModelCheckpoint(c.path_to_models+'/model', save_best_only=True),
    TensorBoard(c.path_to_summaries)
    ]

model = facial_recognizer(c)
model.compile(
    # optimizer=SGD(lr=1e-3, momentum=0.9),
    optimizer='adam',
    loss={'out_au': 'binary_crossentropy',
          'out_emotion': tools.masked_loss},
    metrics={'out_au': 'accuracy',
             'out_emotion': tools.masked_acc})

train_gen, test_gen = tools.get_generators(c)
# train_gen, test_gen = tools.fake_generator(c), tools.fake_generator(c)
model.fit_generator(generator=train_gen,
                    steps_per_epoch=200,
                    epochs=c.epochs,
                    verbose=1,
                    callbacks=call_backs,
                    validation_data=test_gen,
                    validation_steps=50,
                    max_queue_size=c.max_queue_size,
                    workers=c.workers)
model.save(c.path_to_models+'/model_OE')