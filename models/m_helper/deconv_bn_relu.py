from tensorflow.keras import layers
from tensorflow import keras

class deconv_bn_relu(keras.layers.Layer):
    def __init__(self, num_filters, kernel_size=3, strides=1, name=None, trainable=True):
        super(deconv_bn_relu, self).__init__(name=name, trainable=trainable)
        self.deconv = layers.Conv2DTranspose(num_filters, kernel_size, strides, padding='same')
        self.bn = layers.BatchNormalization()
        self.relu = layers.ReLU()  
    def call(self,input):
        x = self.deconv(input)
        x = self.bn(x)
        x = self.relu(x)
        return x