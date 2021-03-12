
import tensorflow as tf
import numpy as np
from PIL import Image
from utils.utils import imsShow
import time
import random
import matplotlib.pyplot as plt

# def plot_dset_one(model, dset, i_img=0):
#     '''
#     visualize one img and the truth in the tf.data.Dataset.
#     input: tf.data.Dataset, and the image number.
#     '''
#     for (img_high, img_mid, img_low), truth_low in dset:
#         pre = model([img_high, img_mid, img_low], training=False)
#         # pre = tf.where(pre>0.5, 1, 0)
#         img_high, img_mid, img_low, truth_low, pre = img_high.numpy(), img_mid.numpy(),\
#                                             img_low.numpy(), truth_low.numpy(), pre.numpy()
#         figure = imsShow([img_high[i_img], img_mid[i_img], img_low[i_img], truth_low[i_img], pre[i_img]],\
#                             ['img_high', 'img_mid', 'img_low', 'truth_low','prediction'],[2,2,2,0,0], \
#                             [[2,1,0], [2,1,0], [2,1,0], [0,0,0], [0,0,0]], figsize=(20,4))
#     return figure

def plot_dset_one(model, dset, i_img=0):
    '''
    visualize one img and the truth in the tf.data.Dataset.
    input: tf.data.Dataset, and the image number.
    '''
    for (img_high, img_mid, img_low), truth_low in dset:
        pre,weight_high,weight_mid,weight_low = model([img_high, img_mid, img_low], training=False)
        # pre = tf.where(pre>0.5, 1, 0)
        img_high, img_mid, img_low, truth_low, pre = img_high.numpy(), img_mid.numpy(),\
                                            img_low.numpy(), truth_low.numpy(), pre.numpy()
        figure = imsShow([img_high[i_img], img_mid[i_img], img_low[i_img], truth_low[i_img], pre[i_img]],\
                            ['img_high', 'img_mid', 'img_low', 'truth_low','prediction'],[2,2,2,0,0], \
                            [[2,1,0], [2,1,0], [2,1,0], [0,0,0], [0,0,0]], figsize=(20,4))
        print('weight_high:{:f}, weight_mid:{:f}, weight_low:{:f}'.format(weight_high[i_img].numpy().squeeze(), \
                                    weight_mid[i_img].numpy().squeeze(), weight_low[i_img].numpy().squeeze()))
        plt.show()
    return figure


def fig2tensor(figure):
    """
    Converts the matplotlib plot specified by 'figure' to tf.Tensor data.
    """
    figure.canvas.draw()
    w,h = figure.canvas.get_width_height()
    buf = np.frombuffer(figure.canvas.tostring_rgb(), dtype=np.uint8)
    buf.shape=(w,h,3)
    buf = np.roll(buf,3,axis=2)
    img = Image.frombytes("RGB",(w,h),buf.tostring())
    img_tf = tf.convert_to_tensor(np.array(img))
    return tf.expand_dims(img_tf, 0)

## metrics
class miou_binary(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.where(y_pred>0.5, 1, 0)
        super().update_state(y_true, y_pred, sample_weight)

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
    def call(self,y_true, y_pred):
        H, W, C = y_true.get_shape().as_list()[1:]
        smooth = 1e-5
        pred_flat = tf.reshape(y_pred, [-1, H * W * C])
        true_flat = tf.reshape(y_true, [-1, H * W * C])
        intersection = 2 * tf.reduce_sum(pred_flat * true_flat, axis=1) + smooth
        denominator = tf.reduce_sum(pred_flat, axis=1) + tf.reduce_sum(true_flat, axis=1) + smooth
        loss = 1 - tf.reduce_mean(intersection / denominator)
        return loss

### callback functions
class img_vis_callback(tf.keras.callbacks.Callback):
    def __init__(self, dset, i_img='False', i_max=1):
        '''
        i_img: the number order of the one batch of the dset.
        '''
        super(img_vis_callback, self).__init__()
        self.dset = dset
        self.i_img = i_img
        self.i_max = i_max
    def on_epoch_end(self, epoch, logs=None):
        if self.i_img is 'False':
            self.i_img = random.randint(0, self.i_max)
            fig2tensor(plot_dset_one(self.model, dset = self.dset, i_img = self.i_img))
            self.i_img = 'False'
        else:
            fig2tensor(plot_dset_one(self.model, dset = self.dset, i_img = self.i_img))

class time_record_callback(tf.keras.callbacks.Callback):
    def __init__(self):
        super(time_record_callback,self).__init__()
        self.start = time.time()
    def on_epoch_end(self, epoch, logs=None):
        if epoch > 0:
            print('Time consuming: {:2f}'.format(time.time()-self.start))
            self.start = time.time()

class lr_scheduler_callback(tf.keras.callbacks.Callback):
    """Learning rate scheduler which sets the learning rate according to schedule.
      Arguments:
      schedule function: a function that takes an epoch index and current learning rate
                as inputs and returns a new learning rate as output (float).
    """
    def __init__(self, schedule):
        super(lr_scheduler_callback, self).__init__()
        self.schedule = schedule

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, "lr"):
            raise ValueError('Optimizer must have a "lr" attribute.')
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        scheduled_lr = self.schedule(epoch, lr)
        tf.keras.backend.set_value(self.model.optimizer.lr, scheduled_lr)
        # print("\nEpoch %d: Learning rate is %6.4f." % (epoch, scheduled_lr))

class stop_min_loss_callback(tf.keras.callbacks.Callback):

    """Stop training when the loss is at its min, i.e. the loss stops decreasing.
  Arguments:
      patience: Number of epochs to wait after min has been hit. After this
      number of no improvement, training stops.
  """
    def __init__(self, patience=20):
        super(stop_min_loss_callback, self).__init__()
        self.patience = patience
        # best_weights to store the weights at which the minimum loss occurs.
        self.best_weights = None

    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get("loss")
        if np.less(current, self.best):
            self.best = current
            self.wait = 0
            # Record the best weights if current results is better (less).
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print("Restoring model weights from the end of the best epoch.")
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %d: early stopping" % (self.stopped_epoch + 1))

class best_model_save_callback(tf.keras.callbacks.Callback):
    """save the best model during model training
  Arguments:
      metric_name: the selected accuracy metric
      path_model: the path for the model saving.
  """
    def __init__(self, metric_name='val_MIoU', path_model_weight='logs/best_model'):
        super(best_model_save_callback, self).__init__()
        self.metric_name = metric_name
        self.path_model_weight = path_model_weight
    def on_train_begin(self, logs=None):
        self.best = -np.Inf
    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.metric_name)
        if np.greater(current, self.best):
            self.best = current
            self.best_weights = self.model.get_weights()
            self.model.save_weights(self.path_model_weight+'/model_epoch_{:d}'.format(epoch+1))