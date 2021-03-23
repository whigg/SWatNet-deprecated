import sys
sys.path.append("/home/yons/Desktop/developer-luo/SWatNet/utils")
from imgShow import imsShow
import matplotlib.pyplot as plt
import tensorflow as tf

def plot_dset_one(model, dset, i_img=0, binary=True, weight=True):
    '''
    visualize one img and the truth in the tf.data.Dataset.
    input: tf.data.Dataset, and the image number.
    '''
    for (img_high, img_mid, img_low), truth_low in dset:
        if weight:
            pre, weight_low, weight_mid, weight_high = model([img_high, img_mid, img_low], training=False)
        else:
            pre = model([img_high, img_mid, img_low], training=False)
        if binary:
            pre = tf.where(pre>0.5, 1, 0)
        img_high, img_mid, img_low, truth_low, pre = img_high.numpy(), img_mid.numpy(),img_low.numpy(),\
                                    truth_low.numpy(),pre.numpy()
        figure = imsShow([img_high[i_img], img_mid[i_img], img_low[i_img], truth_low[i_img], pre[i_img]],\
                ['img_high', 'img_mid', 'img_low', 'truth_low','pre'],[2,2,2,0,0], \
                            [[2,1,0], [2,1,0], [2,1,0], [0,0,0], [0,0,0]], figsize=(20,4))
        if weight:
            print('weight_high: {:f}, weight_mid: {:f}, weight_low: {:f}'.format(weight_high[i_img].numpy().squeeze(), \
                            weight_mid[i_img].numpy().squeeze(), weight_low[i_img].numpy().squeeze()))
        plt.show()
    return figure

