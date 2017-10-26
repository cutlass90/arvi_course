from tools import Config

# Notes for self:
# audio signal must be scaled from -1 to 1
# ASSUME THAT 1 - IS TRUE SAMPLE AND 0 - IS FAKE SAMPLE

config = Config(
    batch_size = 1,
    audio_size = 2**17, #8 sec
    text_size = 64,
    n_compress_block = 8,
    convo_size = 8, # initial number of filters in convo
    char_to_class = {' ': 0,
                     'a': 1,
                     'b': 2,
                     'c': 3,
                     'd': 4,
                     'e': 5,
                     'f': 6,
                     'g': 7,
                     'h': 8,
                     'i': 9,
                     'j': 10,
                     'k': 11,
                     'l': 12,
                     'm': 13,
                     'n': 14,
                     'o': 15,
                     'p': 16,
                     'q': 17,
                     'r': 18,
                     's': 19,
                     't': 20,
                     'u': 21,
                     'v': 22,
                     'w': 23,
                     'x': 24,
                     'y': 25,
                     'z': 26,
                     '0': 27,
                     '1': 28,
                     '2': 29,
                     '3': 30,
                     '4': 31,
                     '5': 32,
                     '6': 33,
                     '7': 34,
                     '8': 35,
                     '9': 36,
                     }

    )

