import sys
sys.path.append("/home/yons/Desktop/developer-luo/SWatNet/models")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.models as models
from m_helper.convert_g_l import convert_g_l
from base_models.mobilenet import mobilenet_v2
from m_helper._conv_block import conv_bn_relu, deconv_bn_relu

def upsize(tensor, size_target):  
    '''bilinear upsampling'''
    y = tf.image.resize(images=tensor, size=size_target)
    return y

def model_triple(input_shape=(256,256,4), base_model=mobilenet_v2, \
                    dfea_layer=88, mfea_layer=21, lfea_layer=8, \
                    scale_high=2048, scale_mid=512, scale_low=256, \
                    nclass=2):
    '''
    description: unet structure with backbone network, 
                 the backbone provides 3-layers feature.
    '''
    encoder_high = base_model(input_shape, nclass)
    encoder_mid = base_model(input_shape, nclass)
    encoder_low = base_model(input_shape, nclass)
    fea_layers = [lfea_layer, mfea_layer, dfea_layer]
    '''-----feature encoding-----
    feas -> [lfea, mfea, dfea]->[downx2, downx4, downx16]
    '''
    feas_high, feas_mid, feas_low = [],[],[]
    '''---high-scale feature learning---'''
    '''!!!!! Note: learning by one model or learning by three model'''
    for layer in fea_layers:
        x_high = encoder_high.get_layer(index=layer).output
        x_high = conv_bn_relu(filters=128, ksize=3, strides=1)(x_high)
        feas_high.append(x_high)

    '''---mid-scale feature learning---'''
    for layer in fea_layers:
        x_mid = encoder_mid.get_layer(index=layer).output
        x_mid = conv_bn_relu(filters=128, ksize=3, strides=1)(x_mid)
        feas_mid.append(x_mid)

    '''---low-scale feature learning---'''
    for layer in fea_layers:
        x_low = encoder_low.get_layer(index=layer).output 
        x_low = conv_bn_relu(filters=128, ksize=3, strides=1)(x_low)
        feas_low.append(x_low)

    '''-----feature decoding-----
        high-scale, mid-scale and low-scale
    '''
    feas_scales = [feas_high, feas_mid, feas_low]
    feas_final = []     # -> [high-scale, mid-scale, low-scale]
    for feas in feas_scales:
        '''---deep fea -> mid fea---'''
        feas_deep_up = upsize(tensor=feas[2], \
                            size_target=[scale_low//4, scale_low//4])
        feas_d_m = layers.concatenate([feas_deep_up, feas[1]])
        feas_d_m = conv_bn_relu(filters=256, ksize=3, strides=1)(feas_d_m)
        '''---deep&mid -> low---'''
        feas_dm_up = upsize(tensor=feas_d_m, \
                            size_target=[scale_low//2, scale_low//2])
        feas_dml = layers.concatenate([feas_dm_up, feas[0]])
        feas_dml = conv_bn_relu(filters=256, ksize=3, strides=1)(feas_dml)
        feas_final.append(feas_dml)     # -> [high-scale, mid-scale, low-scale]
    
    '''-----scale convert-----'''
    feas_high2low = convert_g_l(img_g=feas_final[0], global_size=scale_high, \
                                        local_size=scale_low)
    feas_mid2low = convert_g_l(img_g=feas_final[1], global_size=scale_mid, \
                                        local_size=scale_low)
    
    '''-----concat-----'''
    feas_concat = layers.concatenate([feas_high2low, feas_mid2low, \
                                            feas_final[2]], name='feas_concat')

    '''-----output layer-----'''
    x_oupt = deconv_bn_relu(filters=512, ksize=3, strides=2)(feas_concat)
    x_oupt = layers.Conv2D(filters=1, kernel_size=1, strides=1, \
                                                activation='sigmoid')(x_oupt)

    model = models.Model(inputs=[encoder_low.input, encoder_mid.input, \
                    encoder_high.input], outputs=x_oupt, name='mobilenetv2_triple')

    return model

def model_triple_2(input_shape=(256,256,4), base_model=mobilenet_v2, \
                        dfea_layer=88, mfea_layer=21, lfea_layer=8, \
                        scale_high=2048,scale_mid=512, scale_low=256,\
                        nclass=2):
    '''
    description: unet structure with backbone network, 
                 the backbone provides 3-layers feature.
    '''
    input_high = keras.Input(shape=input_shape)
    input_mid = keras.Input(shape=input_shape)
    input_low = keras.Input(shape=input_shape)
    encoder = base_model(input_shape, nclass)
    encoder_body = encoder._layers.pop(0)   # remove the top input layer

    fea_layers = [lfea_layer, mfea_layer, dfea_layer]

    '''-----feature encoding-----
    feas -> [lfea, mfea, dfea]->[downx2, downx4, downx16]
    '''
    feas_high, feas_mid, feas_low = [],[],[]
    '''---high-scale feature learning---'''
    '''!!!!! Note: learning by one model or learning by three model'''
    _ = encoder_body(input_high)
    for layer in fea_layers:
        x_high = encoder_body.get_layer(index=layer-1).output
        x_high = conv_bn_relu(filters=128, ksize=3, strides=1)(x_high)
        feas_high.append(x_high)

    '''---mid-scale feature learning---'''
    _ = encoder_body(input_mid)
    for layer in fea_layers:
        x_mid = encoder_body.get_layer(index=layer-1).output
        x_mid = conv_bn_relu(filters=128, ksize=3, strides=1)(x_mid)
        feas_mid.append(x_mid)

    '''---low-scale feature learning---'''
    _ = encoder_body(input_low)
    for layer in fea_layers:
        x_low = encoder_body.get_layer(index=layer-1).output 
        x_low = conv_bn_relu(filters=128, ksize=3, strides=1)(x_low)
        feas_low.append(x_low)

    '''-----feature decoding-----
        high-scale, mid-scale and low-scale
    '''
    feas_scales = [feas_high, feas_mid, feas_low]
    feas_final = []     # -> [high-scale, mid-scale, low-scale]
    for feas in feas_scales:
        '''---deep fea -> mid fea---'''
        feas_deep_up = upsize(tensor=feas[2], \
                            size_target=[scale_low//4, scale_low//4])
        feas_d_m = layers.concatenate([feas_deep_up, feas[1]])
        feas_d_m = conv_bn_relu(filters=256, ksize=3, strides=1)(feas_d_m)
        '''---deep&mid -> low---'''
        feas_dm_up = upsize(tensor=feas_d_m, \
                            size_target=[scale_low//2, scale_low//2])
        feas_dml = layers.concatenate([feas_dm_up, feas[0]])
        feas_dml = conv_bn_relu(filters=256, ksize=3, strides=1)(feas_dml)
        feas_final.append(feas_dml)     # -> [high-scale, mid-scale, low-scale]
    
    '''-----scale convert-----'''
    feas_high2low = convert_g_l(img_g=feas_final[0], global_size=scale_high, \
                                        local_size=scale_low)
    feas_mid2low = convert_g_l(img_g=feas_final[1], global_size=scale_mid, \
                                        local_size=scale_low)
    
    '''-----concat-----'''
    feas_concat = layers.concatenate([feas_high2low, feas_mid2low, \
                                            feas_final[2]], name='feas_concat')

    '''-----output layer-----'''
    x_oupt = deconv_bn_relu(filters=512, ksize=3, strides=2)(feas_concat)
    x_oupt = layers.Conv2D(filters=1, kernel_size=1, strides=1, \
                                                activation='sigmoid')(x_oupt)

    model = models.Model(inputs=[x_high, x_mid, x_low], \
                                outputs=x_oupt, name='mobilenetv2_triple')

    return model

# model = model_triple(input_shape=(256,256,4), base_model=mobilenet_v2)
# model.summary()
