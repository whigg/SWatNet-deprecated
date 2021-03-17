from tensorflow.keras import layers
import tensorflow as tf

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

