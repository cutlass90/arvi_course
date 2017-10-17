from tools import Config

config = Config(
    # data config
    path_to_train_data='./facs',
    test_size = 0.1,
    batch_size = 2,############################################
    img_height = 48,############################################
    img_width = 48,###############################################
    n_emotions = 7,
    n_action_units = 65, # target action units
    landmark_size = 136, # number of features that provide dlib
    img_shape = (640, 490, 3), # original image shape
    n_frames = 10, # number of images in sequence
    # train config
    path_to_summaries = './summaries',
    path_to_log = './log.csv',
    path_to_models = './models',
    epochs = 100,
    max_queue_size = 100,
    workers = 1
    )

