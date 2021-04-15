import sys
sys.path.append("/home/yons/Desktop/developer-luo/SWatNet/models")

import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras.models as models
import tensorflow.keras.backend as backend
from base_models.mobilenet import mobilenet_v2
from m_helper._conv_block import conv_bn_relu, deconv_bn_relu

def upsample(tensor, size_target):
    '''bilinear upsampling'''
    y = tf.image.resize(images=tensor, size=size_target)
    return y

def ASPP(tensor):

    '''atrous spatial pyramid pooling'''
    dims = backend.int_shape(tensor)
    '''---1x1 conv, rate: 1 '''
    y_1 = conv_bn_relu(filters=128, ksize=1, strides=1, \
                        dilation_rate=1, name='aspp_conv2d_rate_1')(tensor)

    '''---3x3 dilated conv, rate: 6'''
    y_6 = conv_bn_relu(filters=128, ksize=3, strides=1, \
                        dilation_rate=6, name='aspp_conv2d_rate_6')(tensor)

    '''---3x3 dilated conv, rate: 12'''
    y_12 = conv_bn_relu(filters=128, ksize=3, strides=1, \
                        dilation_rate=12, name='aspp_conv2d_rate_12')(tensor)
    
    '''---3x3 dilated conv, rate: 18'''
    y_18 = conv_bn_relu(filters=128, ksize=3, strides=1, \
                        dilation_rate=18, name='aspp_conv2d_rate_18')(tensor)

    '''---pooling'''
    y_pool = layers.AveragePooling2D(pool_size=(
                        dims[1], dims[2]), name='average_pooling')(tensor)
    y_pool = conv_bn_relu(filters=128, ksize=1, strides=1)(y_pool)
    y_pool = upsample(tensor=y_pool, size_target=[dims[1], dims[2]])

    '''concat'''
    y = layers.concatenate([y_1, y_6, y_12, y_18, y_pool], name='ASPP_concat')
    y = conv_bn_relu(filters=128, ksize=1, strides=1)(y)
    return y

def deeplabv3_plus(input_shape, base_model, d_feat=80, m_feat=20, l_feat=8, nclasses=2):
    (patch_height, patch_width, patch_channel) = input_shape
    base_model = base_model(input_shape, nclasses)

    '''encoding'''
    '''---deep feature'''
    x_deep = base_model.get_layer(index=d_feat).output  # size:down x16
    x_deep = conv_bn_relu(filters=256, ksize=3, strides=1)(x_deep)
    x_deep_aspp = ASPP(x_deep)
    '''---mid feature'''
    x_mid = base_model.get_layer(index=m_feat).output # size: down x4
    x_mid = conv_bn_relu(filters=256, ksize=3, strides=1)(x_mid)
    '''---low feature'''
    x_low = base_model.get_layer(index=l_feat).output  # size: down x2
    x_low = conv_bn_relu(filters=256, ksize=3, strides=1)(x_low)

    '''decoding'''
    '''---deep -> mid'''
    x_deep_up = upsample(tensor=x_deep_aspp, size_target=[patch_height//4, patch_width//4])
    x_d_m = layers.concatenate([x_deep_up, x_mid], name='concat_deep_mid')
    x_d_m = conv_bn_relu(filters=256, ksize=3, strides=1)(x_d_m)
    '''---deep&mid -> low'''
    x_dm_up = upsample(tensor=x_d_m, size_target=[patch_height//2, patch_width//2])
    x_dm_l = layers.concatenate([x_dm_up, x_low], name='concat_deep_mid_low')
    x_dm_l = conv_bn_relu(filters=512, ksize=3, strides=1)(x_dm_l)
    '''---output layer'''
    x_oupt = deconv_bn_relu(filters=512, ksize=3, strides=2)(x_dm_l)
    x_oupt = layers.Conv2D(filters=1, kernel_size=1, \
                            strides=1, activation='sigmoid')(x_oupt)
    model = models.Model(inputs=base_model.input, outputs=x_oupt, name='unet_mobilenetv2')
    return model

# model = deeplabv3_plus(input_shape=[256,256,4], base_model=mobilenet_v2,nclasses=2)
# model.summary()
