"""
author: xin luo, date: 2021.3.13
descriptions: tensorflow callback functions
"""
import sys
sys.path.append("/home/yons/Desktop/developer-luo/SWatNet/trainer/tra_helper")
import tensorflow as tf
import numpy as np
import time
import random
from plot_dset_one import plot_dset_one
from fig2tensor import fig2tensor

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