from keras.applications.resnet50 import ResNet50
from keras.callbacks import EarlyStopping, ProgbarLogger, ModelCheckpoint
from keras.callbacks import LearningRateScheduler, CSVLogger, TensorBoard
import numpy as np

import tools

batch_size = 2
epochs=25



def mock_generator():
    for i in range(100000):
        img = np.random.random([batch_size, 224, 224, 3])
        targets = np.zeros([batch_size, 1000])
        targets[:, 0] = 1
        yield (img, targets)

train_gen = mock_generator()


model = ResNet50()
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])

def lr_scheduler(current_epoch):
    edge = epochs // 3
    if current_epoch < current_epoch:
        return tools.linear_decay(1e-5, 1e-3, edge, current_epoch)
    else:
        return tools.linear_decay(1e-3, 1e-5, epochs, current_epoch)



call_backs =[
    EarlyStopping('categorical_crossentropy', min_delta=1e-5, patience=2),
    ProgbarLogger('steps'),
    ModelCheckpoint('./models/model', save_best_only=True),
    LearningRateScheduler(lr_scheduler),
    CSVLogger('./log.csv'),
    TensorBoard('./summaries')
]

model.fit_generator(generator=train_gen,
                    steps_per_epoch=10,
                    epochs=epochs,
                    verbose=1,
                    callbacks=call_backs,
                    validation_data=train_gen,
                    validation_steps=10,
                    class_weight=None,
                    max_queue_size=10,
                    workers=5,
                    use_multiprocessing=True,
                    initial_epoch=0)

