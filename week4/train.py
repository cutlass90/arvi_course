import os

from keras.callbacks import EarlyStopping, ProgbarLogger, ModelCheckpoint
from keras.callbacks import LearningRateScheduler, CSVLogger, TensorBoard
from keras.layers import Input
from keras.optimizers import RMSprop
from keras.models import load_model, Model
import numpy as np

import tools
from resnet50 import ResNetGenerator, ResNetDiscriminator
from config import config as c



def get_models():
    discriminator = ResNetDiscriminator(c)
    discriminator.compile(loss={'out_chars':tools.categorical_crossentropy_gan,
                                'true_fake_many':'binary_crossentropy',
                                'true_fake_1':'binary_crossentropy'},
                          optimizer='adam')
    
    generator = ResNetGenerator(c)

    input_signal = Input(shape=(c.audio_size,))
    fake_signal = generator(input_signal)
    discriminator.trainable = False
    out_chars, true_fake_many, true_fake_1 = discriminator(fake_signal)
    combined = Model(inputs=input_signal, outputs=[out_chars,
                                                   true_fake_many,
                                                   true_fake_1])
    combined.compile(loss=['categorical_crossentropy',
                           'binary_crossentropy',
                           'binary_crossentropy'],
                     optimizer='adam')
    return discriminator, generator, combined

def train_disc_step(discriminator, generator, real_signal, fake_signal, real_text):
    # we need that discriminator gives ones for real signal:
    out_chars = real_text
    true_fake_many = np.ones([c.batch_size, 256])
    true_fake_1 = np.ones([c.batch_size, 1])
    # AND
    # we need that discriminator gives zeros for fake signal:
    out_chars = np.vstack([out_chars, real_text])
    true_fake_many = np.vstack([true_fake_many, np.zeros([c.batch_size, 256])])
    true_fake_1 = np.vstack([true_fake_1, np.zeros([c.batch_size, 1])])

    imitate_signal = generator.predict(fake_signal)
    signals = np.vstack([real_signal, imitate_signal])
    discriminator.train_on_batch(signals, [out_chars, true_fake_many, true_fake_1])

def train_gen_step(combined, discriminator, fake_signal):
    # we need that generator give ones for fake signal:
    true_fake_many = np.ones([c.batch_size, 256])
    true_fake_1 = np.ones([c.batch_size, 1])
    out_chars, _, _ = discriminator.predict(fake_signal)
    combined.train_on_batch(fake_signal, [out_chars, true_fake_many, true_fake_1])

def train(discriminator, generator, combined, train_steps=2000):
    gen = tools.fake_generator(c)
    for i in range(train_steps):
        print('iteration', i)
        real_signal, fake_signal, real_text = next(gen)
        train_disc_step(discriminator, generator, real_signal, fake_signal, real_text)
        train_gen_step(combined, discriminator, fake_signal)


def main():
    discriminator, generator, combined = get_models()
    train(discriminator, generator, combined, train_steps=2000)



if __name__ == '__main__':
    main()