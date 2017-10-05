from tools import Config

data_config = Config(
    path_to_data='/mnt/course/autoria',
    valid_size = 0.1,
    batch_size = 64,
    img_height = 224,
    img_width = 224
)

train_config = Config(
    lr_min = 1e-5,
    lr_max = 1e-3,
    n_fozen_layers = [5, 10, 15, 20],
    path_to_summaries = './summaries',
    path_to_log = './log.csv',
    path_to_models = './models',
    epochs = 100,
    max_queue_size = 100,
    workers = 10
)

config = Config(
    scope = 'classifier',
    data = data_config,
    train = train_config,
)
