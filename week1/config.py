from tools import Config

data_config = Config(
    path_to_data='/mnt/course/autoria',
    valid_size = 0.1,
    batch_size = 16,
    img_height = 224,
    img_width = 224
)

train_config = Config(
    epochs = 100,
    max_queue_size = 10,
    workers = 5
)

config = Config(
    scope = 'classifier',
    data = data_config,
    train = train_config,
)
