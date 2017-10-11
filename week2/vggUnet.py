from keras.models import *
from keras.layers import *
from keras.utils.data_utils import get_file

WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'

def Interp(x, shape):
    from keras.backend import tf as ktf
    new_height, new_width = shape
    resized = ktf.image.resize_images(x, [new_height, new_width], align_corners=True)
    return resized

def VGGUnet(c):
    img_input = Input(shape=(c.img_height, c.img_width, 3))
    vgg_level = 3

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    f1 = x # 240 x 200
    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    f2 = x# 120 x 100
    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    f3 = x# 60 x 50
    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    f4 = x# 30 x 25
    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    f5 = x
    vgg = Model(img_input, x)
    weights_path = get_file('vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5',
                            WEIGHTS_PATH_NO_TOP,
                            cache_subdir='models')
    vgg.load_weights(weights_path, by_name=True)
    for layer in vgg.layers:
        layer.trainable = False

    levels = [f1, f2, f3, f4, f5]
    print(levels)

    d = levels[vgg_level]

    d = Conv2D(512, (3, 3), padding='same')(d)
    d = BatchNormalization()(d)
    d = Activation('relu')(d)# 30 x 25

    d = UpSampling2D((2, 2))(d) # 60 x 50
    d = Concatenate(axis=-1)([levels[vgg_level-1], d])
    d = Conv2D(256, (3, 3), padding='same')(d)
    d = BatchNormalization()(d)
    d = Activation('relu')(d)

    d = UpSampling2D((2, 2))(d)# 120 x 100
    d = Concatenate(axis=-1)([levels[vgg_level - 2], d])
    d = Conv2D(128, (3, 3), padding='same')(d)
    d = BatchNormalization()(d)
    d = Activation('relu')(d)

    d = UpSampling2D((2, 2))(d) # 240 x 200
    d = Concatenate(axis=-1)([levels[vgg_level - 3], d])
    d = Conv2D(64, (3, 3), padding='same')(d)
    d = BatchNormalization()(d)
    d = Activation('relu')(d)

    # d = UpSampling2D((2, 2))(d) # 480 x 400
    # d = Concatenate(axis=-1)([levels[vgg_level - 4], d])
    # d = Conv2D(64, (3, 3), padding='same')(d)
    # d = BatchNormalization()(d)
    # d = Activation('relu')(d)

    d = Conv2D(c.n_classes, (1, 1))(d)
    d = Lambda(Interp, arguments={'shape': (c.img_height, c.img_width)})(d)
    d = Activation('sigmoid')(d)

    finalmodel = Model(inputs=img_input, outputs=d)
    finalmodel.summary()

    return finalmodel
