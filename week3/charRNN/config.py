from tools import Config

config = Config(
    path_to_texts='./OE',
    test_size = 0.1,
    batch_size = 1024,
    max_len = 100,


    #train config
    path_to_summaries = './summaries',
    path_to_log = './log.csv',
    path_to_models = './models',
    epochs = 200,
    max_queue_size = 100,
    workers = 1
    )

