import os

from keras.applications.resnet50 import ResNet50
from keras.callbacks import EarlyStopping, ProgbarLogger, ModelCheckpoint
from keras.callbacks import LearningRateScheduler, CSVLogger, TensorBoard
import numpy as np

import tools
from config import config



# def mock_generator():
#     while True:
#         img = np.random.random([config.data.batch_size, 224, 224, 3])
#         targets = np.zeros([config.data.batch_size, 1000])
#         targets[:, 0] = 1
#         yield (img, targets)

# train_gen = mock_generator()


# model = ResNet50()
# model.compile(optimizer='adam',
#               loss='categorical_crossentropy',
#               metrics=['categorical_accuracy'])

# call_backs =[
#     EarlyStopping('categorical_crossentropy', min_delta=1e-5, patience=2),
#     ProgbarLogger('steps'),
#     ModelCheckpoint('./models/model', save_best_only=True),
#     LearningRateScheduler(lambda x: tools.lr_scheduler(x, config.train.epochs)),
#     CSVLogger('./log.csv'),
#     TensorBoard('./summaries')
# ]

# model.fit_generator(generator=train_gen,
#                     steps_per_epoch=config.train.steps_per_epoch,
#                     epochs=config.train.epochs,
#                     verbose=1,
#                     callbacks=call_backs,
#                     validation_data=train_gen,
#                     validation_steps=config.train.validation_steps,
#                     max_queue_size=config.train.max_queue_size,
#                     workers=config.train.workers,
#                     use_multiprocessing=True)

def main():
        # sub_dirs = [x[0] for x in gfile.Walk(image_dir)]
    num_classes = len(os.listdir(config.data.path_to_data))
    print("Number of classes found: {}".format(num_classes))
    print("Using validation percent of %{}".format(validation_pct))
    image_lists = create_image_lists(image_dir, validation_pct)

    generators = get_generators(image_lists, image_dir)

    # Get and train the top layers.
    model = get_top_layer_model(model)
    model = train_model(model, epochs=10, generators=generators)

    # Get and train the mid layers.
    model = get_mid_layer_model(model)
    _ = train_model(model, epochs=100, generators=generators,
                    callbacks=[checkpointer, early_stopper, tensorboard])

    # save model
    model.save('./output/model.hdf5', overwrite=True)


if __name__ == '__main__':
    main()