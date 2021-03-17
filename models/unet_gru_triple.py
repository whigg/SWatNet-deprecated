'''
author: xin luo
date: 2021.3.13
'''

import sys
sys.path.append('/home/yons/Desktop/developer-luo/SWatNet/models')
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
from m_helper.convert_g_l import convert_g_l
from m_helper.resample import dsample, upsample
from m_helper.gru_module import gru_module
from m_helper.conv_bn_relu import conv_bn_relu

class unet_gru_triple(keras.Model):
    ''' 
        description: a improved unet model based on unet_triple.
        feature: we use gru module to determine the importance (weight) of specific scale.
    '''
    def __init__(self, scale_high=2048, scale_mid=512, scale_low=256, nclass=2, 
                 trainable_gru=True, trainable_unet=True, **kwargs):
        super(unet_gru_triple, self).__init__(**kwargs)
        self.trainable_gru = trainable_gru
        self.nclass = nclass
        self.scale_high, self.scale_mid, self.scale_low = scale_high, scale_mid, scale_low
        self.encoder = [
            dsample(exp_channels=64,out_channels=64,scale=2,name='down_1_x2',trainable=trainable_unet),  # 1/2
            dsample(exp_channels=64,out_channels=64,scale=2,name='down_2_x2',trainable=trainable_unet),  # 1/4
            dsample(exp_channels=128,out_channels=64,scale=2,name='down_3_x2',trainable=trainable_unet),  # 1/8
            dsample(exp_channels=128,out_channels=128,scale=4,name='down_4_x4',trainable=trainable_unet), # 1/32
            dsample(exp_channels=256,out_channels=128,scale=4,name='down_5_x4',trainable=trainable_unet), # 1/128
            ]
        self.decoder = [
            upsample(out_channels=128, scale=4, name='up_1_x4',trainable=trainable_unet),
            upsample(out_channels=64, scale=4, name='up_2_x4',trainable=trainable_unet),
            upsample(out_channels=64, scale=2, name='up_3_x2',trainable=trainable_unet),
            upsample(out_channels=64, scale=2, name='up_4_x2',trainable=trainable_unet),
            ]
        self.gru_modules = [gru_module(num_fea=128,name='gru_module_1',trainable = trainable_gru),
                    gru_module(num_fea=128,name='gru_module_2',trainable = trainable_gru),
                    gru_module(num_fea=128,name='gru_module_3',trainable = trainable_gru),
                    gru_module(num_fea=128,name='gru_module_4',trainable = trainable_gru)]
        self.scale_high2low = convert_g_l(global_size=self.scale_high, local_size=self.scale_low)
        self.scale_mid2low = convert_g_l(global_size=self.scale_mid, local_size=self.scale_low)
        self.up_last = upsample(out_channels=64, scale=2, name='last_up_x2',trainable=trainable_unet)
        self.last_conv = conv_bn_relu(num_filters=64, kernel_size=3, strides=1,
                                                name='last_conv', trainable=trainable_unet)
        self.last_outp = layers.Conv2D(1, 1, strides=1, padding='same',
                                    activation='sigmoid', name='output_layer',trainable=trainable_unet)        

    def call(self, inputs):
        x_high, x_mid, x_low = inputs[0], inputs[1], inputs[2]
        ################################
        ## feature encoding
        skips_high, skips_mid, skips_low = [],[],[]
        x_encode_high, x_encode_mid, x_encode_low = x_high, x_mid, x_low
        # low-level feature learning
        for encode_low in self.encoder:
            x_encode_low = encode_low(x_encode_low)
            skips_low.append(x_encode_low)
        skips_low = reversed(skips_low[:-1])
        ################################
        # mid-level feature learning
        for encode_mid in self.encoder:
            x_encode_mid = encode_mid(x_encode_mid)
            skips_mid.append(x_encode_mid)
        skips_mid = reversed(skips_mid[:-1])
        #################################
        # high-level feature learning
        for encode_high in self.encoder:
            x_encode_high = encode_high(x_encode_high)
            skips_high.append(x_encode_high)
        skips_high = reversed(skips_high[:-1])
        ## features concatenation
        x_encode_feas = tf.keras.layers.Concatenate(axis=3)([x_encode_high, x_encode_mid, x_encode_low])
        ##################################
        ## decoding
        x_decode = x_encode_feas
        for i, (gru_m, decode, skip_high, skip_mid, skip_low) in enumerate(zip(self.gru_modules, self.decoder, skips_high, skips_mid, skips_low)):            
            # scale transfer
            skip_high2low = self.scale_high2low(g_img=skip_high)
            skip_mid2low = self.scale_mid2low(g_img=skip_mid)
            # gru: feature weights
            fea_weights = gru_m(cnn_fea_low=skip_low, cnn_fea_mid=skip_mid, cnn_fea_high=skip_high)
            fea_weights = tf.expand_dims(fea_weights, 3)
            # weights normalization
            fea_weights_sum = tf.expand_dims(tf.reduce_sum(fea_weights,1),1)
            fea_weights = tf.divide(fea_weights, fea_weights_sum/3)
            if self.trainable_gru == False:
                fea_weights = tf.ones_like(fea_weights)
            # skip connection
            skip_low = tf.multiply(skip_low, fea_weights[:,0:1,:,:])
            skip_mid2low = tf.multiply(skip_mid2low, fea_weights[:,1:2,:,:])
            skip_high2low = tf.multiply(skip_high2low, fea_weights[:,2:3,:,:])
            x_decode = decode(x_decode)
            x_decode = tf.keras.layers.Concatenate(axis=3)([x_decode, skip_high2low, skip_mid2low, skip_low])
            # x_decode = tf.keras.layers.Add()([x_decode, skip_high2low, skip_mid2low, skip_low])
        ##################################
        #### last processing
        x_decode = self.up_last(x_decode)
        x_decode = self.last_conv(x_decode)
        oupt = self.last_outp(x_decode)
        return oupt, fea_weights[:,0:1,:,:], fea_weights[:,1:2,:,:], fea_weights[:,2:3,:,:]
