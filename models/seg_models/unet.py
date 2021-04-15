import sys
sys.path.append("/home/yons/Desktop/developer-luo/SWatNet/models")

from tensorflow.keras import layers
from tensorflow import keras
from m_helper._sample_block import dsample, upsample

class unet(keras.Model):
    ''' 
        Description: unet model for single-scale image processing
        Integrate the multi-scale features for surface water mapping
        the input global image should be down sampled to same to the local image.
    '''
    def __init__(self, nclass=2, **kwargs):
        super(unet, self).__init__(**kwargs)
        self.nclass = nclass
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
        if self.nclass == 2:
            self.last = keras.Sequential([layers.Conv2D(1, 1, strides=1,
                        padding='same', activation='sigmoid')], name='last_conv')
        else:
            self.last = keras.Sequential([layers.Conv2D(self.nclass, 1, strides=1, 
                        padding='same', activation='softmax')], name='last_conv')
    def call(self, input):
        x_encode = input
        '''feature encoding''' 
        skips = []
        for encode in self.encoder:
            x_encode = encode(x_encode)
            skips.append(x_encode)
        skips = reversed(skips[:-1])
        '''feature decoding'''
        x_decode = x_encode
        for i, (decode, skip) in enumerate(zip(self.decoder, skips)):
            x_decode = decode(x_decode)
            x_decode = layers.Concatenate(name='concat_%d'%(i))([x_decode, skip])
        output = self.up_last(x_decode)
        output = self.last(output)
        return output
