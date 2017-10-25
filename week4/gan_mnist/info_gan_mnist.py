import os

import matplotlib
matplotlib.use('Agg')
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

from keras.models import Model
from keras.layers import Dense, Activation, Flatten, Reshape, Input
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D, Concatenate
from keras.layers import LeakyReLU, Dropout
from keras.layers import BatchNormalization
from keras.optimizers import RMSprop

import matplotlib.pyplot as plt

BATCH_SIZE = 256
NOISE_SIZE = 100
mnist = input_data.read_data_sets("mnist", one_hot=True)
# ASSUME THAT 1 - IS TRUE SAMPLE AND 0 - IS FAKE SAMPLE

# discriminator
def discriminator_model():
    _inp = Input(shape=(28, 28, 1))
    x = Conv2D(64, 5, strides=2, padding='same')(_inp)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.4)(x)

    x = Conv2D(128, 5, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.4)(x)

    x = Conv2D(256, 5, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.4)(x)

    x = Conv2D(512, 5, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.4)(x)

    x = Flatten()(x)
    true_or_fake = Dense(1, activation='sigmoid', name='true_or_fake')(x)
    out_class = Dense(10, activation='softmax', name='out_class')(x)

    return Model(inputs=_inp, outputs=[true_or_fake, out_class])


def generator_model():
    inp_class = Input(shape=(10,), name='inp_class')
    inp_noise = Input(shape=(NOISE_SIZE,),  name='inp_noise')
    
    x = Concatenate()([inp_class, inp_noise])
    x = Dense(7 * 7 * 256)(x)
    x = BatchNormalization(momentum=0.9)(x)
    x = Activation('relu')(x)
    x = Reshape((7, 7, 256))(x)
    x = Dropout(0.4)(x)

    x = UpSampling2D()(x)
    x = Conv2DTranspose(128, 5, padding='same')(x)
    x = BatchNormalization(momentum=0.9)(x)
    x = Activation('relu')(x)

    x = UpSampling2D()(x)
    x = Conv2DTranspose(64, 5, padding='same')(x)
    x = BatchNormalization(momentum=0.9)(x)
    x = Activation('relu')(x)

    x = Conv2DTranspose(32, 5, padding='same')(x)
    x = BatchNormalization(momentum=0.9)(x)
    x = Activation('relu')(x)

    x = Conv2DTranspose(1, 5, padding='same')(x)
    x = Activation('sigmoid')(x)

    return Model(inputs=[inp_class, inp_noise], outputs=x)

def get_models():
    discriminator = discriminator_model()
    discriminator.compile(loss={'true_or_fake': 'binary_crossentropy',
                            'out_class': 'categorical_crossentropy'},
                        optimizer=RMSprop(lr=0.0002, decay=6e-8),
                        metrics=['accuracy'])

    generator = generator_model()
    generator.compile(loss='binary_crossentropy',
                        optimizer=RMSprop(lr=0.0002, decay=6e-8),
                        metrics=['accuracy'])

    noise = Input(shape=(NOISE_SIZE,))
    labels = Input(shape=(10,))
    fake_imgs = generator([labels, noise])
    discriminator.trainable = False
    true_or_fake, class_ = discriminator(fake_imgs)
    combined = Model([labels, noise], [true_or_fake, class_])
    combined.compile(loss=['binary_crossentropy', 'categorical_crossentropy'],
                     optimizer=RMSprop(lr=0.0002, decay=6e-8),
                     metrics=['accuracy'])
    return discriminator, generator, combined

def train_disc_step(discriminator, generator):
    # we need that discriminator give ones for real images,
    # and predict correct class:
    imgs, labels = mnist.train.next_batch(BATCH_SIZE)
    imgs = imgs.reshape(BATCH_SIZE, 28, 28, 1)
    discriminator.train_on_batch(imgs, [np.ones([BATCH_SIZE]), labels])
    # AND
    # we need that discriminator give zeros for fake images
    # and don't care about it's class, but it is impusible in Keras =/
    # that's why we use labels
    noise = np.random.uniform(-1.0, 1.0, size=[BATCH_SIZE, NOISE_SIZE])
    fake_imgs = generator.predict([labels, noise])
    discriminator.train_on_batch(fake_imgs, [np.zeros([BATCH_SIZE]), labels])

def train_gen_step(combined):
    # we need that generator give ones for fake images
    # and generate images of correct class:
    noise = np.random.uniform(-1.0, 1.0, size=[BATCH_SIZE, NOISE_SIZE])
    _, labels = mnist.train.next_batch(BATCH_SIZE)
    combined.train_on_batch([labels, noise], [np.ones([BATCH_SIZE]), labels])
    
def train(discriminator, generator, combined, train_steps=2000, save_interval=50):
    noise_input = np.random.uniform(-1.0, 1.0, size=[16, NOISE_SIZE])
    for i in range(train_steps):
        train_disc_step(discriminator, generator)
        train_gen_step(combined)
        if (i + 1) % save_interval == 0:
            labels = np.vstack([np.eye(10), np.eye(10)[:6]])
            plot_images(generator, save2file=True, samples=noise_input.shape[0],
                        noise=noise_input, labels=labels, step=(i + 1))


def plot_images(generator, save2file=False, fake=True, samples=16, noise=None,
                labels=True, step=0):
    os.makedirs('./pics', exist_ok=True)
    filename = 'mnist.png'
    if fake:
        if noise is None:
            noise = np.random.uniform(-1.0, 1.0, size=[samples, 100])
        else:
            filename = "./pics/mnist_%d.png" % step
        images = generator.predict([labels, noise])

    plt.figure(figsize=(10, 10))
    for i in range(images.shape[0]):
        plt.subplot(4, 4, i+1)
        image = images[i, :, :, :]
        image = np.reshape(image, [28, 28])
        plt.imshow(image, cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    if save2file:
        plt.savefig(filename)
        plt.close('all')
    else:
        plt.show()

def main():
    discriminator, generator, combined = get_models()
    train(discriminator, generator, combined)


if __name__ == '__main__':
    main()
