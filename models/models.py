
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf

class _conv_bn_relu(keras.layers.Layer):
    def __init__(self, num_filters, kernel_size=3, strides=1, name=None, trainable=True):
        super(_conv_bn_relu, self).__init__(name=name, trainable=trainable)
        self.conv = layers.Conv2D(num_filters, kernel_size, strides=strides, padding='same')
        self.bn = layers.BatchNormalization()
        self.relu = layers.ReLU()
    def call(self,input):
        x = self.conv(input)
        x = self.bn(x)
        x = self.relu(x)
        return x


class _deconv_bn_relu(keras.layers.Layer):
    def __init__(self, num_filters, kernel_size=3, strides=1, name=None, trainable=True):
        super(_deconv_bn_relu, self).__init__(name=name, trainable=trainable)
        self.deconv = layers.Conv2DTranspose(num_filters, kernel_size, strides, padding='same')
        self.bn = layers.BatchNormalization()
        self.relu = layers.ReLU()  
    def call(self,input):
        x = self.deconv(input)
        x = self.bn(x)
        x = self.relu(x)
        return x

class _dwconv_bn_relu(keras.layers.Layer):
    def __init__(self, kernel_size=3, strides=1, depth=1, name=None, trainable=True):
        super(_dwconv_bn_relu, self).__init__(name=name, trainable=trainable)
        self.dwconv = layers.DepthwiseConv2D(kernel_size, strides, depth_multiplier=depth, padding="same")
        self.bn = layers.BatchNormalization()
        self.relu = layers.ReLU()
    def call(self,input):
        x = self.dwconv(input)
        x = self.bn(x)
        x = self.relu(x)
        return x


def h_sigmoid(x):
    return tf.nn.relu6(x + 3) / 6

# class SEBlock(tf.keras.layers.Layer):
#     def __init__(self, input_channels, r=16):
#         super(SEBlock, self).__init__()
#         self.pool = tf.keras.layers.GlobalAveragePooling2D()
#         self.fc1 = tf.keras.layers.Dense(units=input_channels // r)
#         self.fc2 = tf.keras.layers.Dense(units=input_channels)

#     def call(self, inputs, **kwargs):
#         branch = self.pool(inputs)
#         branch = self.fc1(branch)
#         branch = tf.nn.relu(branch)
#         branch = self.fc2(branch)
#         branch = h_sigmoid(branch)
#         branch = tf.expand_dims(input=branch, axis=1)
#         branch = tf.expand_dims(input=branch, axis=1)
#         output = inputs * branch
#         return output

# class M_SEBlock(tf.keras.layers.Layer):
#     def __init__(self, input_channels, global_size, local_size, r=16):
#         super(M_SEBlock, self).__init__()
#         self.convert_g_l = convert_g_l(global_size, local_size)
#         self.pool = tf.keras.layers.GlobalAveragePooling2D()
#         self.fc1 = tf.keras.layers.Dense(units=input_channels // r)
#         self.fc2 = tf.keras.layers.Dense(units=input_channels)

#     def call(self, inputs, **kwargs):
#         branch = self.pool(inputs)
#         branch = self.fc1(branch)
#         branch = tf.nn.relu(branch)
#         branch = self.fc2(branch)
#         branch = h_sigmoid(branch)
#         branch = tf.expand_dims(input=branch, axis=1)
#         branch = tf.expand_dims(input=branch, axis=1)
#         inputs_l = self.convert_g_l(g_img=inputs)
#         output = inputs_l * branch
#         return output



class dsample(keras.layers.Layer):
    def __init__(self, exp_channels, out_channels, scale=2, name=None, trainable=True):
        super(dsample, self).__init__(name=name, trainable=trainable)
        self.scale = scale
        self.pool = layers.AveragePooling2D(pool_size=(scale, scale), padding='valid')
        self.conv_bn_relu_in = _conv_bn_relu(num_filters=exp_channels, kernel_size=1, strides = 1)
        self.dwconv_bn_relu_1 = _dwconv_bn_relu(kernel_size=3, strides = 1)
        self.dwconv_bn_relu_2 = _dwconv_bn_relu(kernel_size=3, strides = 1)
        self.conv_bn_relu_out = _conv_bn_relu(num_filters=out_channels, kernel_size=1, strides = 1)
    def call(self, input):
        if self.scale==2:
            x = self.pool(input)
            x = self.conv_bn_relu_in(x)    
            x = self.dwconv_bn_relu_1(x)
            x = self.conv_bn_relu_out(x)
        elif self.scale==4:
            x = self.pool(input)
            x = self.conv_bn_relu_in(x) 
            x = self.dwconv_bn_relu_1(x)
            x = self.dwconv_bn_relu_2(x)
            x = self.conv_bn_relu_out(x)
        return x


class upsample(keras.layers.Layer):
    def __init__(self, out_channels, scale=2, name=None, trainable=True):
        super(upsample, self).__init__(name=name, trainable=trainable)
        self.scale = scale
        self.conv_bn_relu_1 = _conv_bn_relu(num_filters=out_channels, kernel_size=3, strides = 1)
        self.conv_bn_relu_2 = _conv_bn_relu(num_filters=out_channels, kernel_size=3, strides = 1)
        self.dwconv_bn_relu = _dwconv_bn_relu(kernel_size=3, strides = 1, depth=2)
    def call(self, input):
        if self.scale==2:
            x = layers.UpSampling2D(size=2, interpolation='bilinear')(input)
            x = self.conv_bn_relu_1(x)
            x = self.dwconv_bn_relu(x)
        elif self.scale==4:
            x = layers.UpSampling2D(size=4, interpolation='bilinear')(input)
            x = self.conv_bn_relu_1(x)
            x = self.conv_bn_relu_2(x)
            x = self.dwconv_bn_relu(x)
        return x

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

class gru_module(tf.keras.layers.Layer):
    def __init__(self, num_fea=128, name=None, trainable=True):
        super(gru_module,self).__init__(name=name,trainable=trainable)
        self.pool = layers.GlobalAveragePooling2D()
        self.bi_gru_1 = layers.Bidirectional(tf.keras.layers.GRU(num_fea, return_sequences=True),merge_mode='ave')
        # self.bi_gru_2 = layers.Bidirectional(tf.keras.layers.GRU(1, return_sequences=True),merge_mode='ave')
        self.fc = layers.Dense(units=1)
        self.sigmoid = layers.Activation('sigmoid')
    def call(self, cnn_fea_low, cnn_fea_mid, cnn_fea_high, **kwargs):
        x_low, x_mid, x_high = self.pool(cnn_fea_low), self.pool(cnn_fea_mid), self.pool(cnn_fea_high)
        x_low, x_mid, x_high = tf.expand_dims(x_low, 1),tf.expand_dims(x_mid, 1),tf.expand_dims(x_high, 1)
        x_feas_gru = tf.keras.layers.Concatenate(axis=1)([x_low,x_mid,x_high])
        x_outp = self.bi_gru_1(x_feas_gru)
        # x_outp = self.bi_gru_2(x_outp)
        x_outp = self.fc(x_outp)
        x_outp = self.sigmoid(x_outp)
        return x_outp

class unet_module(keras.layers.Layer):
    '''
    the image size is downsampled to 1/64 using encoder module,
    and thun upsampled to the original size using decoder module.
    '''
    def __init__(self, name='unet_module', **kwargs):
        super(unet_module, self).__init__(name=name, **kwargs)
        self.encoder = [
            dsample(exp_channels=32, out_channels=16, scale=2, name='down_1_x2', SE=False),  # 1/2
            dsample(exp_channels=64, out_channels=16, scale=2, name='down_2_x2', SE=False),  # 1/4
            dsample(exp_channels=128, out_channels=32, scale=2, name='down_3_x2', SE=False),  # 1/8
            dsample(exp_channels=128, out_channels=32, scale=4, name='down_4_x4', SE=False), # 1/32
            dsample(exp_channels=256, out_channels=64, scale=4, name='down_5_x4', SE=False), # 1/128
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

class UNet(keras.Model):
    ''' Integrate the multi-scale features for surface water mapping
        the input global image should be down sampled to same to the local image.
    '''
    def __init__(self, nclass=2, **kwargs):
        super(UNet, self).__init__(**kwargs)
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

class UNet_x3(keras.Model):
    ''' Integrate the multi-scale features for surface water mapping
        the input global image should be down sampled to same to the local image.
    '''
    def __init__(self, scale_high=2048, scale_mid=512, scale_low=256, nclass=2, **kwargs):
        super(UNet_x3, self).__init__(**kwargs)
        self.nclass = nclass
        self.scale_high, self.scale_mid, self.scale_low = scale_high, scale_mid, scale_low
        self.unet_module_high = unet_module(name='unet_m_high')
        self.unet_module_mid = unet_module(name='unet_m_mid')
        self.unet_module_low = unet_module(name='unet_m_low')
        self.last_conv_high = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, strides=1, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
        ], name='last_conv_high')
        self.last_conv_mid = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, strides=1, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
        ], name='last_conv_mid')
        self.last_conv_low = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, strides=1, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
        ], name='last_conv_low')
        self.last_high = tf.keras.Sequential([tf.keras.layers.Conv2D(1, 1, strides=1,
                    padding='same', activation='sigmoid')], name='output_layer_high')
        self.last_mid = tf.keras.Sequential([tf.keras.layers.Conv2D(1, 1, strides=1,
                    padding='same', activation='sigmoid')], name='output_layer_mid')
        self.last_low = tf.keras.Sequential([tf.keras.layers.Conv2D(1, 1, strides=1,
                    padding='same', activation='sigmoid')], name='output_layer_low')

    def call(self, inputs):
        x_high, x_mid, x_low = inputs[0], inputs[1], inputs[2]
        ### high feature learning
        x_high, x_high_encode = self.unet_module_high(x_high, training=True)
        ### mid feature learning
        x_mid, x_mid_encode = self.unet_module_mid(x_mid, training=True)
        ### low feature learning
        x_low, x_low_encode = self.unet_module_low(x_low, training=True)
        x_high2low = convert_g_l(global_size=self.scale_high, local_size=self.scale_low)(g_img=x_high)
        x_mid2low = convert_g_l(global_size=self.scale_mid, local_size=self.scale_low)(g_img=x_mid)
        x_high2low = self.last_conv_high(x_high2low)
        x_mid2low = self.last_conv_mid(x_mid2low)
        x_low = self.last_conv_low(x_low)
        oupt_high = self.last_high(x_high2low)
        oupt_mid = self.last_mid(x_mid2low)
        oupt_low = self.last_low(x_low)
        return oupt_high, oupt_mid, oupt_low

class UNet_triple(keras.Model):
    ''' Integrate the multi-scale features for surface water mapping
        the input global image should be down sampled to same to the local image.
    '''
    def __init__(self, scale_high=2048, scale_mid=512, scale_low=256, nclass=2, **kwargs):
        super(UNet_triple, self).__init__(**kwargs)
        self.nclass = nclass
        self.scale_high, self.scale_mid, self.scale_low = scale_high, scale_mid, scale_low
        self.unet_module = unet_module(name='unet_m')
        self.last_conv = tf.keras.Sequential([
            layers.Conv2D(32, 3, strides=1, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
        ], name='last_conv')
        self.last = tf.keras.Sequential([tf.keras.layers.Conv2D(1, 1, strides=1,
                    padding='same', activation='sigmoid')], name='output_layer')

    def call(self, inputs):
        x_high, x_mid, x_low = inputs[0], inputs[1], inputs[2]
        ### high feature learning
        x_high, x_high_encode = self.unet_module(x_high, training=True)
        ### mid feature learning
        x_mid, x_mid_encode = self.unet_module(x_mid, training=True)
        ### low feature learning
        x_low, x_low_encode = self.unet_module(x_low, training=True)
        ### scale transfer
        x_high2low = convert_g_l(global_size=self.scale_high, local_size=self.scale_low)(g_img=x_high)
        x_mid2low = convert_g_l(global_size=self.scale_mid, local_size=self.scale_low)(g_img=x_mid)
        ### features weighting
        x_merge = x_high2low + x_mid2low + x_low
        x_merge = self.last_conv(x_merge)
        oupt = self.last(x_merge)
        return oupt

class UNet_gru_triple(keras.Model):
    ''' Integrate the multi-scale features for surface water mapping
        the input global image should be down sampled to same to the local image.
    '''
    def __init__(self, scale_high=2048, scale_mid=512, scale_low=256, nclass=2, 
                 trainable_gru=True, trainable_unet=True, **kwargs):
        super(UNet_gru_triple, self).__init__(**kwargs)
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
        self.last_conv = _conv_bn_relu(num_filters=64, kernel_size=3, strides=1,
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

class UNet_triple_v2(keras.Model):
    ''' Integrate the multi-scale features for surface water mapping
        the input global image should be down sampled to same to the local image.
    '''
    def __init__(self, scale_high=2048, scale_mid=512, scale_low=256, nclass=2, **kwargs):
        super(UNet_triple_v2, self).__init__(**kwargs)
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
        self.discrimination_low = tf.keras.Sequential(
            [layers.GlobalAveragePooling2D(),
            layers.Dense(units=128, activation='relu'),
            layers.Dense(units=1, activation='sigmoid'),
            ], name='dis_low')
        self.discrimination_mid = tf.keras.Sequential(
            [layers.GlobalAveragePooling2D(),
            layers.Dense(units=128, activation='relu'),
            layers.Dense(units=1, activation='sigmoid'),
            ], name='dis_mid')
        self.scale_high2low = convert_g_l(global_size=self.scale_high, local_size=self.scale_low)
        self.scale_mid2low = convert_g_l(global_size=self.scale_mid, local_size=self.scale_low)

        self.up_last = upsample(out_channels=64, scale=2, name='last_up_x2')
        self.last_conv = _conv_bn_relu(num_filters=64, kernel_size=3, strides=1)
        self.last_outp = tf.keras.Sequential([tf.keras.layers.Conv2D(1, 1, strides=1,
                        padding='same', activation='sigmoid')], name='output_layer')

    def call(self, inputs):
        x_encode_high, x_encode_mid, x_encode_low = inputs[0], inputs[1], inputs[2]
        #### feature encoding 
        skips_high, skips_mid, skips_low = [],[],[]
        #############################
        # low-level feature learning
        for encode_low in self.encoder:
            x_encode_low = encode_low(x_encode_low)
            skips_low.append(x_encode_low)
        skips_low = reversed(skips_low[:-1])
        dis_low = self.discrimination_low(x_encode_low)
        dis_low = tf.expand_dims(tf.expand_dims(dis_low,-1), -1)
        #############################
        # mid-level feature learning
        for encode_mid in self.encoder:
            x_encode_mid = tf.multiply(x_encode_mid, dis_low)
            x_encode_mid = encode_mid(x_encode_mid)
            x_encode_mid2low = self.scale_mid2low(x_encode_mid)
            skips_mid.append(x_encode_mid2low)
        skips_mid = reversed(skips_mid[:-1])
        x_encode_low_mid = tf.keras.layers.Concatenate(name='concat_fea_low_mid')([x_encode_low, x_encode_mid2low])
        dis_mid = self.discrimination_mid(x_encode_low_mid)
        dis_mid = tf.expand_dims(tf.expand_dims(dis_mid,-1),-1)
        #############################
        # high-level feature learning
        for encode_high in self.encoder:
            x_encode_high = tf.multiply(x_encode_high, dis_mid)
            x_encode_high = encode_high(x_encode_high)
            x_encode_high2low = self.scale_high2low(x_encode_high)
            skips_high.append(x_encode_high2low)
        skips_high = reversed(skips_high[:-1])
        
        #############################     
        ##### feature decoding
        x_decode = layers.Concatenate()([x_encode_low, x_encode_mid2low, x_encode_high2low])
        for i, (decode, skip_high, skip_mid, skip_low) in enumerate(zip(self.decoder,skips_high,skips_mid,skips_low)):            
            skip_high = tf.multiply(skip_high, dis_mid)
            skip_mid = tf.multiply(skip_mid, dis_low)
            x_decode = decode(x_decode)
            x_decode = layers.Concatenate()([x_decode, skip_high, skip_mid, skip_low])
        ############################
        #### last processing
        x_decode = self.up_last(x_decode)
        x_decode = self.last_conv(x_decode)
        oupt = self.last_outp(x_decode)
        return oupt