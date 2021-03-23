import sys
sys.path.append("/home/yons/Desktop/developer-luo/SWatNet/models/m_helper")

from tensorflow import keras
import tensorflow as tf
from resample import dsample, upsample

class unet_module(keras.layers.Layer):
    '''
    the image size is downsampled to 1/64 using encoder module,
    and thun upsampled to the original size using decoder module.
    '''
    def __init__(self, name='unet_module', **kwargs):
        super(unet_module, self).__init__(name=name, **kwargs)
        self.encoder = [
            dsample(exp_channels=32, out_channels=16, scale=2, name='down_1_x2'),  # 1/2
            dsample(exp_channels=64, out_channels=16, scale=2, name='down_2_x2'),  # 1/4
            dsample(exp_channels=128, out_channels=32, scale=2, name='down_3_x2'),  # 1/8
            dsample(exp_channels=128, out_channels=32, scale=4, name='down_4_x4'), # 1/32
            dsample(exp_channels=256, out_channels=64, scale=4, name='down_5_x4'), # 1/128
            ]
        self.decoder = [
            upsample(out_channels=64, scale=4, name='up_1_x4'),
            upsample(out_channels=64, scale=4, name='up_2_x4'),
            upsample(out_channels=64, scale=2, name='up_3_x2'),
            upsample(out_channels=32, scale=2, name='up_4_x2'),
        ]
        self.up_last = upsample(out_channels=32, scale=2, name='last_up_x2')

    def call(self, inputs):
        x_encode = inputs
        #### feature encoding 
        skips = []
        for encode in self.encoder:
            x_encode = encode(x_encode)
            skips.append(x_encode)
        skips = reversed(skips[:-1])
        #### feature decoding
        x_decode = x_encode
        for i, (decode, skip) in enumerate(zip(self.decoder, skips)):
            x_decode = decode(x_decode)
            x_decode = tf.keras.layers.Concatenate(name='concat_%d'%(i))([x_decode, skip])
        output = self.up_last(x_decode)
        return output, x_encode