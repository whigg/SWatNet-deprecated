import tensorflow as tf

def convert_g_l(img_g, global_size, local_size):
    '''global_size should be divisible by local_size.
    '''
    scale_dif = global_size//local_size
    size_g = img_g.shape[1]
    size_l = size_g//scale_dif
    if size_l >= 1:
        ''' crop -> enlarge scale '''
        row_l_min = (size_g - size_l)//2
        img_l = tf.image.crop_to_bounding_box(img_g, row_l_min, row_l_min, size_l, size_l)
        img_l = tf.image.resize(img_l, [size_g, size_g], method='nearest')
    else:
        ''' enlarge scale -> crop '''
        row_l_min = (size_g*scale_dif-size_g)//2
        img_l = tf.image.resize(img_g, [size_g*scale_dif, \
                                                size_g*scale_dif], method='nearest')
        img_l = tf.image.crop_to_bounding_box(img_l, row_l_min, row_l_min, size_g, size_g)
    return img_l



