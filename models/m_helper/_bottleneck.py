from re import T
import sys
sys.path.append("/home/yons/Desktop/developer-luo/SWatNet/models/m_helper")
from tensorflow.keras import layers
from _conv_block import conv_bn_relu6, dwconv_bn_relu6


def _bottleneck(inputs, filters, kernel, t, s, r=False):
    tchannel = inputs.shape[-1] * t
    x = conv_bn_relu6(filters=tchannel, ksize=1, strides=1)(inputs)
    x = layers.DepthwiseConv2D(kernel, strides=(s, s), depth_multiplier=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(6.)(x)
    x = layers.Conv2D(filters, (1, 1), strides=(1, 1), padding='same')(x)  # 降维，改层为瓶颈层
    x = layers.BatchNormalization()(x)
    if r:
        x = layers.add([x, inputs])
    return x

class _bottleneck_c(layers.Layer):
    def __init__(self, input_channel, filters, kernel, t, strides, r=False):
        ''' t: multiplier of the middle-layer output channels
            s: strides of the conv2d layer
            r: resiual connect or not.
        '''
        super(_bottleneck_c, self).__init__()
        self.filters = filters
        self.kernel = kernel
        self.t, self.s, self.r = t, strides, r
        self.tchannel = input_channel * t
        self.conv_bn_relu6 = conv_bn_relu6(filters=self.tchannel, ksize=1, strides=1)
        self.dwconv_bn_relu6 = dwconv_bn_relu6(ksize=kernel, strides=strides, depth=1)
        self.conv2D = layers.Conv2D(filters, (1, 1), strides=(1, 1), padding='same')
        self.bn = layers.BatchNormalization()
    def call(self, input):
        x = self.conv_bn_relu6(input)
        x = self.dwconv_bn_relu6(x)
        x = self.conv2D(x)
        x = self.bn(x)
        if self.r:
            x = x+input
        return x