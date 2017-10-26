from config import config as c
from resnet50 import ResNetGenerator, ResNetDiscriminator
import numpy as np
from keras.layers import Input

# model = ResNetGenerator(c)
model = ResNetDiscriminator(c)
a = model(Input(shape=(c.audio_size,)))
# model.summary()