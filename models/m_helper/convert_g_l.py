from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf

class convert_g_l(keras.layers.Layer):
    def __init__(self, global_size, local_size, name=None):
        super(convert_g_l, self).__init__(name=name)
        self.scale_dif = global_size//local_size
    def call(self, g_img):
        height = g_img.shape[1]
        height_g = height*self.scale_dif
        row_g_min = height_g//2-height//2
        x = tf.image.resize(g_img, [height_g, height_g], method='nearest')
        x = tf.image.crop_to_bounding_box(x, row_g_min, row_g_min, height, height)
        return x
