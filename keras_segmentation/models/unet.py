from keras.models import *
from keras.layers import *

from .model_utils import get_segmentation_model
from .mobilenet import get_mobilenet_encoder
from .basic_models import vanilla_encoder

def _unet(n_classes, encoder, l1_skip_conn=True, input_height=416,
          input_width=608, channels=3):

    img_input, levels = encoder(
        input_height=input_height, input_width=input_width, channels=channels)
    [f1, f2, f3, f4, f5] = levels

    o = f4

    o = (ZeroPadding2D((1, 1)))(o)
    o = (Conv2D(512, (3, 3), padding='valid' , activation='relu'))(o)
    o = (BatchNormalization())(o)

    o = (UpSampling2D((2, 2)))(o)
    o = (concatenate([o, f3]))
    o = (ZeroPadding2D((1, 1)))(o)
    o = (Conv2D(256, (3, 3), padding='valid', activation='relu'))(o)
    o = (BatchNormalization())(o)

    o = (UpSampling2D((2, 2)))(o)
    o = (concatenate([o, f2]))
    o = (ZeroPadding2D((1, 1)))(o)
    o = (Conv2D(128, (3, 3), padding='valid' , activation='relu'))(o)
    o = (BatchNormalization())(o)

    o = (UpSampling2D((2, 2)))(o)

    if l1_skip_conn:
        o = (concatenate([o, f1]))

    o = (ZeroPadding2D((1, 1)))(o)
    o = (Conv2D(64, (3, 3), padding='valid', activation='relu', name="seg_feats"))(o)
    o = (BatchNormalization())(o)

    o = Conv2D(n_classes, (3, 3), padding='same')(o)

    model = get_segmentation_model(img_input, o)

    return model


def unet(n_classes, input_height=416, input_width=608, encoder_level=3, channels=3):

    model = _unet(n_classes, vanilla_encoder,
                  input_height=input_height, input_width=input_width, channels=channels)
    model.model_name = "unet"
    return model

def mobilenet_unet(n_classes, input_height=224, input_width=224,
                   encoder_level=3, channels=3):

    model = _unet(n_classes, get_mobilenet_encoder,
                  input_height=input_height, input_width=input_width, channels=channels)
    model.model_name = "mobilenet_unet"
    return model

if __name__ == '__main__':
    m = _unet(101, vanilla_encoder)
    # m = _unet( 101 , get_mobilenet_encoder ,True , 224 , 224)