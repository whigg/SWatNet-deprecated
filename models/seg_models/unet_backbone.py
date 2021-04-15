import sys
sys.path.append("/home/yons/Desktop/developer-luo/SWatNet/models")

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow.keras.models as models
from base_models.mobilenet import mobilenet_v2
from m_helper._conv_block import conv_bn_relu, deconv_bn_relu

def upsize(tensor, size_target):  
    '''bilinear upsampling'''
    y = tf.image.resize(images=tensor, size=size_target)
    return y

def unet_backbone(input_shape, base_model, dfea_layer=88, mfea_layer=21, \
                                                    lfea_layer=8, nclass=2):
    '''
    description: unet structure with backbone network, 
                 the backbone provides 3-layers feature.
    '''
    (patch_height, patch_width, patch_channel) = input_shape
    base_model = base_model(nclass=nclass)
    '''-----encoding-----
       size: [down x16, down x4, down x2]
    '''
    layers_list = [dfea_layer, mfea_layer, lfea_layer]
    feas = []
    for layer in layers_list:
        fea = base_model.get_layer(index=layer).output  
        fea = conv_bn_relu(filters=256, ksize=3, strides=1)(fea)
        feas.append(fea)

    '''-----decoding-----'''
    '''deep -> mid'''
    x_deep_up = upsize(tensor=feas[0], size_target=[patch_height//4, patch_width//4])
    x_d_m = layers.concatenate([x_deep_up, feas[1]], name='concat_deep_mid')
    x_d_m = conv_bn_relu(filters=256, ksize=3, strides=1)(x_d_m)
    '''deep&mid -> low'''
    x_dm_up = upsize(tensor=x_d_m, size_target=[patch_height//2, patch_width//2])
    x_dm_l = layers.concatenate([x_dm_up, feas[2]], name='concat_deep_mid_low')
    x_dm_l = conv_bn_relu(filters=512, ksize=3, strides=1)(x_dm_l)

    '''-----output layer-----'''
    x_oupt = deconv_bn_relu(filters=512, ksize=3, strides=2)(x_dm_l)
    x_oupt = layers.Conv2D(filters=1, kernel_size=1, \
                            strides=1, activation='sigmoid')(x_oupt)
    model = models.Model(inputs=base_model.input, outputs=x_oupt, name='unet_mobilenetv2')
    return model

# model = unet_backbone(input_shape=(256,256,4), base_model=mobilenet_v2, \
#                             dfea_layer=88, mfea_layer=21,lfea_layer=8, nclass=2)
# model.summary()

