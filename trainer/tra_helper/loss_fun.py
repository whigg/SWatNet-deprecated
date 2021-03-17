import tensorflow as tf

## loss function
class FocalLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        alpha=0.5 
        gamma=1
        cla_num = 2
        # label_smoothing=0.05
        # y_true = (1.0-label_smoothing)*y_true + label_smoothing/cla_num
        FL=-alpha*y_true*((1-y_pred)**gamma)*tf.math.log(tf.clip_by_value(y_pred, 1e-8, 1.0))-\
            (1-alpha)*(1.0-y_true)*(y_pred**gamma)*tf.math.log(tf.clip_by_value(1-y_pred, 1e-8, 1.0))
        return tf.math.reduce_mean(FL)

class DiceLoss_2d(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        H, W, C = y_true.get_shape().as_list()[1:]
        smooth = 1e-5
        pred_flat = tf.reshape(y_pred, [-1, H * W * C])
        true_flat = tf.reshape(y_true, [-1, H * W * C])
        intersection = 2 * tf.reduce_sum(pred_flat * true_flat, axis=1) + smooth
        denominator = tf.reduce_sum(pred_flat, axis=1) + tf.reduce_sum(true_flat, axis=1) + smooth
        loss = 1 - tf.reduce_mean(intersection / denominator)
        return loss
