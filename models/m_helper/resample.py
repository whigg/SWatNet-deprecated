import sys
sys.path.append("/home/yons/Desktop/developer-luo/SWatNet/models/m_helper")

from tensorflow.keras import layers
from tensorflow import keras
from conv_bn_relu import conv_bn_relu
from dwconv_bn_relu import dwconv_bn_relu

class dsample(keras.layers.Layer):
    def __init__(self, exp_channels, out_channels, scale=2, name=None, trainable=True):
        super(dsample, self).__init__(name=name, trainable=trainable)
        self.scale = scale
        self.pool = layers.AveragePooling2D(pool_size=(scale, scale), padding='valid')
        self.conv_bn_relu_in = conv_bn_relu(num_filters=exp_channels, kernel_size=1, strides = 1)
        self.dwconv_bn_relu_1 = dwconv_bn_relu(kernel_size=3, strides = 1)
        self.dwconv_bn_relu_2 = dwconv_bn_relu(kernel_size=3, strides = 1)
        self.conv_bn_relu_out = conv_bn_relu(num_filters=out_channels, kernel_size=1, strides = 1)
    def call(self, input):
        if self.scale==2:
            x = self.pool(input)
            x = self.conv_bn_relu_in(x)    
            x = self.dwconv_bn_relu_1(x)
            x = self.conv_bn_relu_out(x)
        elif self.scale==4:
            x = self.pool(input)
            x = self.conv_bn_relu_in(x) 
            x = self.dwconv_bn_relu_1(x)
            x = self.dwconv_bn_relu_2(x)
            x = self.conv_bn_relu_out(x)
        return x

class upsample(keras.layers.Layer):
    def __init__(self, out_channels, scale=2, name=None, trainable=True):
        super(upsample, self).__init__(name=name, trainable=trainable)
        self.scale = scale
        self.conv_bn_relu_1 = conv_bn_relu(num_filters=out_channels, kernel_size=3, strides = 1)
        self.conv_bn_relu_2 = conv_bn_relu(num_filters=out_channels, kernel_size=3, strides = 1)
        self.dwconv_bn_relu = dwconv_bn_relu(kernel_size=3, strides = 1, depth=2)
    def call(self, input):
        if self.scale==2:
            x = layers.UpSampling2D(size=2, interpolation='bilinear')(input)
            x = self.conv_bn_relu_1(x)
            x = self.dwconv_bn_relu(x)
        elif self.scale==4:
            x = layers.UpSampling2D(size=4, interpolation='bilinear')(input)
            x = self.conv_bn_relu_1(x)
            x = self.conv_bn_relu_2(x)
            x = self.dwconv_bn_relu(x)
        return x
