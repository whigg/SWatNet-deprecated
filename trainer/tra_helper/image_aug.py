import tensorflow as tf
import random

#### Data augmentation: noisy, filp, rotate. 
def image_aug(imgs, truth, flip=True, rot=True, noisy=True):
    (img_high, img_mid, img_low) = imgs
    if flip == True:
        if tf.random.uniform(()) > 0.5:
            if random.randint(1,2) == 1:  ## horizontal or vertical mirroring
                img_high = tf.image.flip_left_right(img_high)
                img_mid = tf.image.flip_left_right(img_mid)
                img_low = tf.image.flip_left_right(img_low)
                truth = tf.image.flip_left_right(truth)
            else: 
                img_high = tf.image.flip_up_down(img_high)
                img_mid = tf.image.flip_up_down(img_mid)
                img_low = tf.image.flip_up_down(img_low)
                truth = tf.image.flip_up_down(truth)
    if rot == True:
        if tf.random.uniform(()) > 0.5: 
            degree = random.randint(1,3)
            img_high = tf.image.rot90(img_high, k=degree)
            img_mid = tf.image.rot90(img_mid, k=degree)
            img_low = tf.image.rot90(img_low, k=degree)
            truth = tf.image.rot90(truth, k=degree)
    if noisy == True:
        if tf.random.uniform(()) > 0.5:
            std = random.uniform(0.001, 0.1)
            gnoise = tf.random.normal(shape=tf.shape(img_high), mean=0.0, stddev=std, dtype=tf.float32)
            img_high = tf.add(img_high, gnoise)
            img_mid = tf.add(img_mid, gnoise)
            img_low = tf.add(img_low, gnoise)
    return (img_high, img_mid, img_low), truth