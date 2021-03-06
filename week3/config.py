from tools import Config

config = Config(
    # augmentation
    sequence_change = 0.3,
    zoom_range = (1.2, 1.2),
    random_shear = 0.2,
    random_rotation = 20,
    # data config
    rectangle_imgs = True,
    path_to_data='./facs',
    saved_paths='./paths.pkl',
    # path_to_data='/mnt/course/datasets/facs/',
    test_size = 0.1,
    batch_size = 16,
    img_height = 224,
    img_width = 224,
    n_emotions = 7,
    n_action_units = 41, # target action units
    landmark_size = 2*136, # number of features that provide dlib + delta coding
    n_frames = 10, # number of images in sequence
    # train config
    path_to_summaries = './summaries',
    path_to_log = './log.csv',
    path_to_models = './models',
    epochs = 100,
    max_queue_size = 100,
    workers = 1,
    au_map = {64.0: 40, 1.0: 0, 2.0: 3, 43.0: 33, 4.0: 4, 5.0: 5, 6.0: 6,
              1.5: 1, 9.0: 8, 10.0: 9, 11.0: 10, 12.0: 11, 13.0: 12, 14.0: 13,
              15.0: 14, 16.0: 15, 17.0: 16, 18.0: 17, 1.7: 2, 21.0: 19, 22.0: 20,
              23.0: 21, 24.0: 22, 25.0: 23, 26.0: 24, 27.0: 25, 28.0: 26,
              29.0: 27, 30.0: 28, 31.0: 29, 34.0: 30, 38.0: 31, 39.0: 32,
              7.0: 7, 44.0: 34, 45.0: 35, 54.0: 36, 20.0: 18, 61.0: 37,
              62.0: 38, 63.0: 39},
    emotion_map = {
        0:'Anger',
        1:'Contempt',
        2:'Disgust',
        3:'Fear',
        4:'Happiness',
        5:'Sadness',
        6:'Surprise'
    }
    )

