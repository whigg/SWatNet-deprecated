import sys

from tensorflow.python.keras.engine.sequential import Sequential
sys.path.append("/home/yons/Desktop/developer-luo/SWatNet/models/m_helper")
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
from _conv_block import conv_bn_relu6
from _bottleneck import _bottleneck


def _inverted_residual_block(input, filters, kernel, t, strides, n, name=None):
    '''Built for mobilenet v2'''
    x = _bottleneck(input, filters, kernel, t, strides)
    for i in range(1, n):
        x = _bottleneck(x, filters, kernel, t, 1, True)
    return x

def mobilenet_v2(input_shape=(256,256,4), nclass=2):
    """
    description: layer8-layer21-layer41-layer88-layer99
                 down2 - down2 - down2 -down2  -down2
    # Arguments
        input_shape: An integer or tuple/list of 3 integers, shape
                     of input tensor.
        classes: Integer, number of classes.
    # Returns
        MobileNetv2 model.
    """

    inputs = layers.Input(shape=input_shape)
    x = conv_bn_relu6(filters=32, ksize=3, strides=2)(inputs) # 0.5*size, n_layers = 1
 
    x = _inverted_residual_block(x,16,3,t=1, strides=1, n=1, name='invert_1')  #  n_layers = 3
    x = _inverted_residual_block(x,24,3,t=6, strides=2, n=2, name='invert_1')  # 0.5*size,  n_layers = 6
    x = _inverted_residual_block(x,32,3,t=6, strides=2, n=3, name='invert_1')  # 0.5*size, n_layers = 9
    x = _inverted_residual_block(x,64,3,t=6, strides=2, n=4, name='invert_1')  # 0.5*size, n_layers = 12
    x = _inverted_residual_block(x,96,3,t=6, strides=1, n=3, name='invert_1')  # n_layers = 9
    x = _inverted_residual_block(x,160,3,t=6, strides=2, n=3, name='invert_1')  # 0.5*size, n_layers = 9
    x = _inverted_residual_block(x,320,3,t=6, strides=1, n=1, name='invert_1')  # n_layers = 3

    x = conv_bn_relu6(filters=1280, ksize=1, strides=1)(x)   # n_layers = 1
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Reshape((1, 1, 1280))(x)
    x = layers.Dropout(0.3, name='Dropout')(x)
    x = layers.Conv2D(nclass, (1, 1), padding='same')(x)  # n_layers = 1
    x = layers.Activation('softmax', name='final_activation')(x)
    output = layers.Reshape((nclass,), name='output')(x)
    model = keras.models.Model(inputs, output)

    return model

# model = mobilenet_v2(nclass=2)
# model.summary()

