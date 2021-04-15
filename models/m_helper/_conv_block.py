from tensorflow.keras import layers
from tensorflow import keras

class conv_bn_relu(keras.layers.Layer):
    def __init__(self, filters, ksize=3, strides=1, dilation_rate=1, name=None, trainable=True):
        super(conv_bn_relu, self).__init__(name=name, trainable=trainable)
        self.conv = layers.Conv2D(filters, ksize, strides=strides, \
                                    dilation_rate=dilation_rate, padding='same')
        self.bn = layers.BatchNormalization()
        self.relu = layers.ReLU()
    def call(self,input,training=True):
        x = self.conv(input)
        x = self.bn(x,training=training)
        x = self.relu(x)
        return x

# def conv_bn_relu(inputs, filters, ksize=3, strides=1, dilation_rate=1):
#     x = layers.Conv2D(filters, ksize, strides=strides, \
#                         dilation_rate=dilation_rate, padding='same')(inputs)
#     x = layers.BatchNormalization()(x)
#     x = layers.ReLU()(x)
#     return x



class conv_bn_relu6(keras.layers.Layer):
    '''for MobileNet v2'''
    def __init__(self, filters, ksize=3, strides=1, dilation_rate=1, name=None, trainable=True):
        super(conv_bn_relu6, self).__init__(name=name, trainable=trainable)
        self.conv = layers.Conv2D(filters, ksize, strides=strides, \
                                dilation_rate=dilation_rate, padding='same')
        self.bn = layers.BatchNormalization()
        self.relu6 = layers.ReLU(6.)
    def call(self, input):
        x = self.conv(input)
        x = self.bn(x)
        x = self.relu6(x)
        return x

class deconv_bn_relu(keras.layers.Layer):
    def __init__(self, filters, ksize=3, strides=1, name=None, trainable=True):
        super(deconv_bn_relu, self).__init__(name=name, trainable=trainable)
        self.deconv = layers.Conv2DTranspose(filters, ksize, strides, padding='same')
        self.bn = layers.BatchNormalization()
        self.relu = layers.ReLU()  
    def call(self,input):
        x = self.deconv(input)
        x = self.bn(x)
        x = self.relu(x)
        return x

class dwconv_bn_relu6(keras.layers.Layer):
    '''for MobileNet v2'''
    def __init__(self, ksize=3, strides=1, depth=1, name=None, trainable=True):
        super(dwconv_bn_relu6, self).__init__(name=name, trainable=trainable)
        self.dwconv = layers.DepthwiseConv2D(ksize, strides, depth_multiplier=depth, padding="same")
        self.bn = layers.BatchNormalization()
        self.relu6 = layers.ReLU(6.)
    def call(self,input):
        x = self.dwconv(input)
        x = self.bn(x)
        x = self.relu6(x)
        return x

class dwconv_bn_relu(keras.layers.Layer):
    def __init__(self, ksize=3, strides=1, depth=1, name=None, trainable=True):
        super(dwconv_bn_relu, self).__init__(name=name, trainable=trainable)
        self.dwconv = layers.DepthwiseConv2D(ksize, strides, depth_multiplier=depth, padding="same")
        self.bn = layers.BatchNormalization()
        self.relu = layers.ReLU()
    def call(self,input):
        x = self.dwconv(input)
        x = self.bn(x)
        x = self.relu(x)
        return x