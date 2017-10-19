from keras_vggface.vggface import VGGFace
from keras.engine import  Model
from keras.layers import Flatten, Dense, Input, TimeDistributed, Lambda, LSTM
from keras.layers import Concatenate
from keras.models import Sequential

from config import config as c
def facial_recognizer(c):
    # inputs
    img_inputs = Input(shape=(c.n_frames, c.img_height, c.img_width, 1),
                    name="img_inputs")
    landmark_inputs = Input(shape=(c.n_frames, c.landmark_size), name="landmark_inputs")

    # image feature extractor
    def get_feature_extractor():
        vgg_model = VGGFace(include_top=False,
                            input_shape=(c.img_height, c.img_width, 1),
                            pooling='avg', weights=None)
        for layer in vgg_model.layers:
            layer.trainable = False
        last_layer = vgg_model.get_layer('pool5')
        
        vgg_feature_extractor = Sequential()
        vgg_feature_extractor.add(last_layer)
        vgg_feature_extractor.add(Flatten())
        vgg_feature_extractor.add(Dense(4096, activation='elu'))
        vgg_feature_extractor.add(Dense(2048, activation='elu'))
        return vgg_feature_extractor

    vgg_feature_extractor = get_feature_extractor()
    img_features = TimeDistributed(vgg_feature_extractor)(img_inputs)

    # landmark feature extractor
    landmark_encoder = Sequential()
    landmark_encoder.add(Dense(256, activation='elu', input_shape=(c.landmark_size,)))
    landmark_encoder.add(Dense(128, activation='elu'))
    landmark_features = TimeDistributed(landmark_encoder)(landmark_inputs)

    all_features = Concatenate(axis=-1)([img_features, landmark_features])
    lstm_out = LSTM(256)(all_features)

    out_emotion = Dense(c.n_emotions, activation='softmax', name='out_emotion')(lstm_out)
    out_au = Dense(c.n_action_units, activation='sigmoid', name='out_au')(lstm_out)

    model = Model(inputs=[img_inputs, landmark_inputs], outputs=[out_emotion, out_au])

    return model