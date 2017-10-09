from tools import Config

data_config = Config(
    path_to_data='/mnt/course/datasets/portraits/imgs',
    path_to_masks='/mnt/course/datasets/portraits/masks',
    test_size = 0.1,
    batch_size = 16,
    img_height = 480, #after resize
    img_width = 400, # after resize
    image_shape = (800, 600, 3) # original
)

train_config = Config(
    lr_min = 1e-5,
    lr_max = 1e-3,
    n_fozen_layers = [5, 15, 25],
    path_to_summaries = './summaries',
    path_to_log = './log.csv',
    path_to_models = './models',
    epochs = 10,
    max_queue_size = 100,
    workers = 1
)

config = Config(
    scope = 'classifier',
    data = data_config,
    train = train_config,
)

