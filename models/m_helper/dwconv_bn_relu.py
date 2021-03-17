from tensorflow.keras import layers
from tensorflow import keras

class dwconv_bn_relu(keras.layers.Layer):
    def __init__(self, kernel_size=3, strides=1, depth=1, name=None, trainable=True):
        super(dwconv_bn_relu, self).__init__(name=name, trainable=trainable)
        self.dwconv = layers.DepthwiseConv2D(kernel_size, strides, depth_multiplier=depth, padding="same")
        self.bn = layers.BatchNormalization()
        self.relu = layers.ReLU()
    def call(self,input):
        x = self.dwconv(input)
        x = self.bn(x)
        x = self.relu(x)
        return x
