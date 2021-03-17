# import sys
# sys.path.append('/home/yons/Desktop/developer-luo/SWatNet')
import tensorflow as tf
from tra_helper.miou_binary import miou_binary
from tra_helper.loss_fun import FocalLoss, DiceLoss_2d
import datetime as dt

## super-parameter for data loader
BATCH_SIZE = 8
BUFFER_SIZE = 200
epoches = 200
root_dir = '/home/yons/Desktop/developer-luo/SWatNet'
path_savedmodel = root_dir + '/models/temporal'
tra_scene_file = root_dir + '/data/traScene.tfrecords'
eva_patch_file = root_dir + '/data/evaPatch_256_512_2048_50x.tfrecords'

# metrics during model training
tra_oa = tf.keras.metrics.BinaryAccuracy('tra_oa')
tra_miou = miou_binary(num_classes=2,name='tra_miou')
tra_loss_tracker = tf.keras.metrics.Mean(name="tra_loss")
test_oa = tf.keras.metrics.BinaryAccuracy('test_oa')
test_miou = miou_binary(num_classes=2,name='test_miou')
test_loss_tracker = tf.keras.metrics.Mean(name="test_loss")

# configuration for model training
## optimizer
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=10,
    decay_rate=0.95)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
## loss function
focal_loss = FocalLoss()
binary_ce_loss = tf.keras.losses.BinaryCrossentropy()
dice_loss = DiceLoss_2d()

# Define the Keras TensorBoard
## log path
current_time = (dt.datetime.utcnow()+dt.timedelta(hours=8)).strftime("%Y%m%d-%H%M%S")
train_log_dir = "logs/tensorb/" + current_time + '/train'
test_log_dir = "logs/tensorb/" + current_time + '/test'
## tensorboard writer
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)






