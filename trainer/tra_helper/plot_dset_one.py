import sys
sys.path.append("/home/yons/Desktop/developer-luo/SWatNet/utils")
from imgShow import imsShow
import matplotlib.pyplot as plt
import tensorflow as tf

def plot_dset_one(model, dset, i_patch=0, binary=True, multiscale=True, weight=True):
    '''
    author: xin luo, date: 2021.3.24
    visualize one img and the truth in the tf.data.Dataset.
    input: 
        model and dataset, and the patch number of one batch.
        binary: control the visualized image is a binary image or not
        multiscale: control the input is multi-scale patches of single-scale patches.
        weight: control whether the output contains multi-scale weights or not.
    '''
    for (patch_high, patch_mid, patch_low), truth_low in dset:
        if multiscale:
            if weight:
                pre, weight_low, weight_mid, weight_high = model([patch_high, patch_mid, patch_low], training=False)
            else:
                pre = model([patch_high, patch_mid, patch_low], training=False)
        else:
            pre = model(patch_low)
        if binary:
            pre = tf.where(pre>0.5, 1, 0)
        patch_high, patch_mid, patch_low, truth_low, pre = patch_high.numpy(), patch_mid.numpy(), patch_low.numpy(),\
                                    truth_low.numpy(),pre.numpy()
        figure = imsShow([patch_high[i_patch], patch_mid[i_patch], patch_low[i_patch], truth_low[i_patch], pre[i_patch]],\
                ['patch_high', 'patch_mid', 'patch_low', 'truth_low','pre'],[2,2,2,0,0], \
                            [[2,1,0], [2,1,0], [2,1,0], [0,0,0], [0,0,0]], figsize=(20,4))
        if weight:
            print('weight_high: {:f}, weight_mid: {:f}, weight_low: {:f}'.format(weight_high[i_patch].numpy().squeeze(), \
                            weight_mid[i_patch].numpy().squeeze(), weight_low[i_patch].numpy().squeeze()))
        plt.show()
    return figure

