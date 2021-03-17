import sys
sys.path.append("/home/yons/Desktop/developer-luo/SWatNet/models")

from tensorflow import keras
import tensorflow as tf
from m_helper import unet_module

class unet(keras.Model):
    ''' 
        Description: unet model for single-scale image processing
        Integrate the multi-scale features for surface water mapping
        the input global image should be down sampled to same to the local image.
    '''
    def __init__(self, nclass=2, **kwargs):
        super(unet, self).__init__(**kwargs)
        self.nclass = nclass
        self.unet_module = unet_module(name='unet_m')
        self.last_conv = tf.keras.Sequential([
                    tf.keras.layers.Conv2D(32, 3, strides=1, padding='same'),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.ReLU(),
                    ], name='last_conv')
        if self.nclass == 2:
            self.last = tf.keras.Sequential([tf.keras.layers.Conv2D(1, 1, strides=1,
                        padding='same', activation='sigmoid')], name='last_conv')
        else:
            self.last = tf.keras.Sequential([tf.keras.layers.Conv2D(self.nclass, 1, strides=1, 
                        padding='same', activation='softmax')], name='last_conv')
    def call(self, inputs):
        x = inputs[1]
        x_fea,_ = self.unet_module(x)
        x_fea = self.last_conv(x_fea)
        x_oupt = self.last(x_fea)
        return x_oupt