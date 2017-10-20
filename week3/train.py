import os

from keras.callbacks import EarlyStopping, ProgbarLogger, ModelCheckpoint
from keras.callbacks import LearningRateScheduler, CSVLogger, TensorBoard
from keras.optimizers import RMSprop
from keras.models import load_model

import tools
from facial_recognizer import facial_recognizer
from config import config as c

def get_model():
    model = facial_recognizer(c)
    model.compile(
        # optimizer=SGD(lr=1e-3, momentum=0.9),
        optimizer='adam',
        loss={'out_au': 'binary_crossentropy',
            'out_emotion': tools.masked_loss},
        metrics={'out_au': 'accuracy',
                'out_emotion': tools.masked_acc})
    if os.path.isfile(os.path.join(c.path_to_models, 'model')):
        model.load_weights(os.path.join(c.path_to_models, 'model'))
                        #    custom_objects={'masked_loss': tools.masked_loss})
        print('Model loaded')
    return model

os.makedirs(c.path_to_summaries, exist_ok=True)
os.makedirs(c.path_to_models, exist_ok=True)

model = get_model()

call_backs =[
    # EarlyStopping('val_acc', min_delta=1e-5, patience=20),
    ProgbarLogger('steps'),
    ModelCheckpoint(c.path_to_models+'/model', save_best_only=True),
    TensorBoard(c.path_to_summaries)
    ]

train_gen, test_gen = tools.get_generators(c)
# train_gen, test_gen = tools.fake_generator(c), tools.fake_generator(c)
model.fit_generator(generator=train_gen,
                    steps_per_epoch=50,
                    epochs=c.epochs,
                    verbose=1,
                    callbacks=call_backs,
                    validation_data=test_gen,
                    validation_steps=10,
                    max_queue_size=c.max_queue_size,
                    workers=c.workers)
model.save(c.path_to_models+'/model_OE')