from tools import Config

config = Config(
    path_to_train_imgs='/mnt/course/datasets/coco/train2017',
    path_to_train_json='/mnt/course/datasets/coco/annotations/instances_train2017.json',
    path_to_test_imgs='/mnt/course/datasets/coco/val2017',
    path_to_test_json='/mnt/course/datasets/coco/annotations/instances_val2017.json',
    test_size = 0.1,
    batch_size = 16,
    img_height = 240, #after resize
    img_width = 320, # after resize
    n_classes = 80, # 80 classes

    #train config
    path_to_summaries = './summaries',
    path_to_log = './log.csv',
    path_to_models = './models',
    epochs = 1000,
    max_queue_size = 100,
    workers = 1
    )

