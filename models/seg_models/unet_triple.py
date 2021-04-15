'''
author: xin luo
date: 2021.3.13
'''

import sys
sys.path.append("/home/yons/Desktop/developer-luo/SWatNet/models")
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
from m_helper.convert_g_l import convert_g_l
from m_helper._conv_block import conv_bn_relu
from m_helper._sample_block import dsample, upsample


class unet_triple(keras.Model):
    ''' 
        description: a improved unet model based on unet_triple.
        feature: we add a scale weighting module (discrimination) to determine the 
        importance of specific scale.
    '''
    def __init__(self, scale_high=2048, scale_mid=512, scale_low=256, nclass=2, **kwargs):
        super(unet_triple, self).__init__(**kwargs)
        self.nclass = nclass
        self.scale_high, self.scale_mid, self.scale_low = scale_high, scale_mid, scale_low
        self.encoder = [
            dsample(exp_channels=64, out_channels=32, scale=2, name='down_1_x2'),  # 1/2
            dsample(exp_channels=64, out_channels=32, scale=2, name='down_2_x2'),  # 1/4
            dsample(exp_channels=128, out_channels=64, scale=2, name='down_3_x2'),  # 1/8
            dsample(exp_channels=128, out_channels=128, scale=4, name='down_4_x4'), # 1/32
            dsample(exp_channels=256, out_channels=128, scale=4, name='down_5_x4'), # 1/128
            ]
        self.decoder = [
            upsample(out_channels=128, scale=4, name='up_1_x4'),
            upsample(out_channels=64, scale=4, name='up_2_x4'),
            upsample(out_channels=32, scale=2, name='up_3_x2'),
            upsample(out_channels=32, scale=2, name='up_4_x2'),
        ]

        self.up_last = upsample(out_channels=64, scale=2, name='last_up_x2')
        self.out_layer = tf.keras.Sequential([
                        conv_bn_relu(filters=64, ksize=3, strides=1),
                        tf.keras.layers.Conv2D(1, 1, strides=1,
                                            padding='same', activation='sigmoid')
                        ], name='output_layer')

    def call(self, inputs):
        x_encode_high, x_encode_mid, x_encode_low = inputs[0], inputs[1], inputs[2]

        '''-----feature encoding-----'''
        skips_high, skips_mid, skips_low = [],[],[]
        '''---low-level feature learning---'''
        for encode_low in self.encoder:
            x_encode_low = encode_low(x_encode_low)
            skips_low.append(x_encode_low)
        skips_low = reversed(skips_low[:-1])
        '''---mid-level feature learning---'''
        for encode_mid in self.encoder:
            x_encode_mid = encode_mid(x_encode_mid)
            x_encode_mid2low = convert_g_l(img_g=x_encode_mid, \
                            global_size=self.scale_mid,local_size=self.scale_low)
            skips_mid.append(x_encode_mid2low)
        skips_mid = reversed(skips_mid[:-1])
        x_encode_low_mid = tf.keras.layers.Concatenate(name='concat_fea_low_mid')([x_encode_low, x_encode_mid2low])
        '''---high-level feature learning---'''
        for encode_high in self.encoder:
            x_encode_high = encode_high(x_encode_high)
            x_encode_high2low = convert_g_l(img_g=x_encode_high, \
                            global_size=self.scale_high,local_size=self.scale_low)
            skips_high.append(x_encode_high2low)
        skips_high = reversed(skips_high[:-1])

        '''-----feature decoding-----'''
        '''---feature fusion---'''
        x_decode = layers.Concatenate()([x_encode_low, x_encode_mid2low, x_encode_high2low])
        for i, (decode, skip_high, skip_mid, skip_low) in enumerate(zip(self.decoder,skips_high,skips_mid,skips_low)):
            x_decode = decode(x_decode)
            x_decode = layers.Concatenate()([x_decode, skip_high, skip_mid, skip_low])

        '''-----last processing-----'''
        x_decode = self.up_last(x_decode)
        oupt = self.out_layer(x_decode)
        return oupt

class unet_triple_w(keras.Model):
    ''' 
        description: a improved unet model based on unet_triple.
        feature: we add a scale weighting module (discrimination) to determine the 
        importance of specific scale.
    '''
    def __init__(self, scale_high=2048, scale_mid=512, scale_low=256, nclass=2, **kwargs):
        super(unet_triple_w, self).__init__(**kwargs)
        self.nclass = nclass
        self.scale_high, self.scale_mid, self.scale_low = scale_high, scale_mid, scale_low
        '''-----build encoder-decoder-----'''
        self.encoder = [
            dsample(exp_channels=64, out_channels=32, scale=2, name='down_1_x2'),  # 1/2
            dsample(exp_channels=64, out_channels=32, scale=2, name='down_2_x2'),  # 1/4
            dsample(exp_channels=128, out_channels=64, scale=2, name='down_3_x2'),  # 1/8
            dsample(exp_channels=128, out_channels=128, scale=4, name='down_4_x4'), # 1/32
            dsample(exp_channels=256, out_channels=128, scale=4, name='down_5_x4'), # 1/128
            ]
        self.decoder = [
            upsample(out_channels=128, scale=4, name='up_1_x4'),
            upsample(out_channels=64, scale=4, name='up_2_x4'),
            upsample(out_channels=32, scale=2, name='up_3_x2'),
            upsample(out_channels=32, scale=2, name='up_4_x2'),
        ]
        '''-----weight determination layer------'''
        self.discrimination_low = tf.keras.Sequential(
                        [layers.GlobalAveragePooling2D(),
                        layers.Dense(units=128, activation='relu'),
                        layers.Dense(units=1, activation='sigmoid'),
                        ], name='dis_high')
        self.discrimination_mid = tf.keras.Sequential(
                        [layers.GlobalAveragePooling2D(),
                        layers.Dense(units=128, activation='relu'),
                        layers.Dense(units=1, activation='sigmoid'),
                        ], name='dis_mid')

        self.up_last = upsample(out_channels=64, scale=2, name='last_up_x2')
        self.out_layer = tf.keras.Sequential([
                        conv_bn_relu(filters=64, ksize=3, strides=1),
                        tf.keras.layers.Conv2D(1, 1, strides=1,
                                            padding='same', activation='sigmoid')
                        ], name='output_layer')

    def call(self, inputs):
        x_encode_high, x_encode_mid, x_encode_low = inputs[0], inputs[1], inputs[2]
        '''-----feature encoding-----'''
        skips_high, skips_mid, skips_low = [],[],[]
        '''---low-level feature learning---'''
        for encode_low in self.encoder:
            x_encode_low = encode_low(x_encode_low)
            skips_low.append(x_encode_low)
        skips_low = reversed(skips_low[:-1])
        dis_low = self.discrimination_low(x_encode_low)
        dis_low = tf.expand_dims(tf.expand_dims(dis_low,-1), -1)
        '''---mid-level feature learning---'''
        for encode_mid in self.encoder:
            x_encode_mid = tf.multiply(x_encode_mid, dis_low)
            x_encode_mid = encode_mid(x_encode_mid)
            x_encode_mid2low = convert_g_l(img_g=x_encode_mid, \
                            global_size=self.scale_mid,local_size=self.scale_low)
            skips_mid.append(x_encode_mid2low)
        skips_mid = reversed(skips_mid[:-1])
        x_encode_low_mid = tf.keras.layers.Concatenate(name='concat_fea_low_mid')([x_encode_low, x_encode_mid2low])
        dis_mid = self.discrimination_mid(x_encode_low_mid)
        dis_mid = tf.expand_dims(tf.expand_dims(dis_mid,-1),-1)
        '''---high-level feature learning---'''
        for encode_high in self.encoder:
            x_encode_high = tf.multiply(x_encode_high, dis_mid)
            x_encode_high = encode_high(x_encode_high)
            x_encode_high2low = convert_g_l(img_g=x_encode_high, \
                            global_size=self.scale_high,local_size=self.scale_low)
            skips_high.append(x_encode_high2low)
        skips_high = reversed(skips_high[:-1])

        '''-----feature decoding-----'''
        '''---feature fusion---'''
        x_decode = layers.Concatenate()([x_encode_low, \
                                        x_encode_mid2low, x_encode_high2low])
        for i, (decode, skip_high, skip_mid, skip_low) in enumerate(zip(self.decoder,skips_high,skips_mid,skips_low)):            
            skip_high = tf.multiply(skip_high, dis_mid)
            skip_mid = tf.multiply(skip_mid, dis_low)
            x_decode = decode(x_decode)
            x_decode = layers.Concatenate()([x_decode, skip_high, skip_mid, skip_low])
        '''---last processing---'''
        x_decode = self.up_last(x_decode)
        oupt = self.out_layer(x_decode)
        return oupt


# inputs = (tf.ones([1, 256, 256, 4],tf.float32), \
#         tf.ones([1,256,256,4],tf.float32), tf.ones([1,256,256,4],tf.float32))
# model = unet_triple(scale_high=2048, scale_mid=512, scale_low=256)
# oupt = model(inputs, training=False)
# model.summary()