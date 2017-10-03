from tools import Config

data_config = Config(
    path_to_data='/media/nazar/DATA/datasets/TIMIT',
    batch_size = 2
)

train_config = Config(
    epochs = 25,
    steps_per_epoch = 10,
    validation_steps = 5,
    max_queue_size = 10,
    workers = 5
)

config = Config(
    scope = 'classifier',
    data = data_config,
    train = train_config,
)
