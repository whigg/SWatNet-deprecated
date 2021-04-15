import tensorflow as tf
import random


def image_aug(imgs, truth, flip=True, rot=True, noisy=True, missing=False):
    '''
    author: xin luo, date: 2021.3.25
    description: Data augmentation: noisy, filp, rotate and missing.'''

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

def img_missing(x_batch):
    '''
    des: random edge missing for the mid and high-level patches.
    x_batch = (batch_high, batch_mid, batch_low)
    !!note: we can integrate this function to the data_aug function cause we can't assign value (0) to the patch tensor in the graph.
    '''
    batch_high, batch_mid, batch_low = x_batch
    patch_height, patch_width = batch_high.shape[1:3]
    batch_high_miss = batch_high.numpy()
    batch_mid_miss = batch_mid.numpy()
    while tf.random.uniform(()) > 0.5: #
        i_img = random.randint(0, batch_high_miss.shape[0]-1) # num of patch
        while random.randint(1,2) == 1:
            '''up, high-scale'''
            edge_row = random.randint(1, patch_height//4)
            batch_high_miss[i_img, 0:edge_row, :, :] = 0
        while random.randint(1,2) == 1:
            '''up, mid-scale'''
            edge_row = random.randint(1, patch_height//4)
            batch_mid_miss[i_img, 0:edge_row, :, :] = 0
        while random.randint(1,2) == 1:
            '''down, high-scale'''
            edge_row = random.randint(1, patch_height//4)
            batch_high_miss[i_img, -edge_row:, :, :] = 0
        while random.randint(1,2) == 1:
            '''down, mid-scale'''
            edge_row = random.randint(1, patch_height//4)
            batch_mid_miss[i_img, -edge_row:, :, :] = 0
        while random.randint(1,2) == 1:
            '''left, high-scale'''
            edge_col = random.randint(1, patch_height//4)
            batch_high_miss[i_img, :, 0:edge_col, :] = 0
        while random.randint(1,2) == 1:
            '''left, mid-scale'''
            edge_col = random.randint(1, patch_height//4)
            batch_mid_miss[i_img, :, 0:edge_col, :] = 0
        while random.randint(1,2) == 1:
            '''right, high-scale'''
            edge_col = random.randint(1, patch_height//4)
            batch_high_miss[i_img,  -edge_col:, :, :] = 0
        while random.randint(1,2) == 1:
            '''right, mid-scale'''
            edge_col = random.randint(1, patch_height//4)
            batch_mid_miss[i_img,  -edge_col:, :, :] = 0
        batch_high = tf.convert_to_tensor(batch_high_miss)
        batch_mid = tf.convert_to_tensor(batch_mid_miss)
    return (batch_high, batch_mid, batch_low)
