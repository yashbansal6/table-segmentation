from keras.models import *
from keras.layers import *
import keras.backend as K
import keras

BASE_WEIGHT_PATH = ('https://github.com/fchollet/deep-learning-models/'
                    'releases/download/v0.6/')

def relu6(x):
    return K.relu(x, max_value=6)

def conv(inputs, filters, kernel=(3, 3), strides=(1, 1)):

    filters = int(filters)
    x = ZeroPadding2D(padding=(1, 1), name='conv1_pad')(inputs)
    x = Conv2D(filters, kernel, use_bias=False, name='conv1', strides=strides)(x)
    x = BatchNormalization(name='conv1_bn')(x)
    return Activation(relu6, name='conv1_relu')(x)


def depthwiseConv(inputs, pointwise_conv_filters, strides=(1, 1), block_id=1):

    pointwise_conv_filters = int(pointwise_conv_filters)

    x = ZeroPadding2D((1, 1), name='conv_pad_%d' % block_id)(inputs)
    x = DepthwiseConv2D((3, 3), use_bias=False, strides=strides, name='conv_dw_%d' % block_id)(x)
    x = BatchNormalization(name='conv_dw_%d_bn' % block_id)(x)
    x = Activation(relu6, name='conv_dw_%d_relu' % block_id)(x)

    x = Conv2D(pointwise_conv_filters, (1, 1), padding='same',
               use_bias=False, name='conv_pw_%d' % block_id)(x)
    x = BatchNormalization(name='conv_pw_%d_bn' % block_id)(x)
    return Activation(relu6, name='conv_pw_%d_relu' % block_id)(x)


def get_mobilenet_encoder(input_height=224, input_width=224,
                          pretrained='imagenet', channels=3):

    assert input_height % 32 == 0
    assert input_width % 32 == 0

    img_input = Input(shape=(input_height, input_width, channels))

    x = conv(img_input, 32, strides=(2, 2))
    x = depthwiseConv(x, 64, block_id=1)
    f1 = x

    x = depthwiseConv(x, 128, strides=(2, 2), block_id=2)
    x = depthwiseConv(x, 128, block_id=3)
    f2 = x

    x = depthwiseConv(x, 256, strides=(2, 2), block_id=4)
    x = depthwiseConv(x, 256, block_id=5)
    f3 = x

    x = depthwiseConv(x, 512, strides=(2, 2), block_id=6)
    x = depthwiseConv(x, 512, block_id=7)
    x = depthwiseConv(x, 512, block_id=8)
    x = depthwiseConv(x, 512, block_id=9)
    x = depthwiseConv(x, 512, block_id=10)
    x = depthwiseConv(x, 512, block_id=11)
    f4 = x

    x = depthwiseConv(x, 1024, strides=(2, 2), block_id=12)
    x = depthwiseConv(x, 1024, block_id=13)
    f5 = x

    if pretrained == 'imagenet':
        model_name = 'mobilenet_%s_%d_tf_no_top.h5' % ('1_0', 224)

        weight_path = BASE_WEIGHT_PATH + model_name
        weights_path = keras.utils.get_file(model_name, weight_path)

        Model(img_input, x).load_weights(weights_path, by_name=True, skip_mismatch=True)

    return img_input, [f1, f2, f3, f4, f5]
