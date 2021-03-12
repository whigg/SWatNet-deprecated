#!/usr/bin/env python
# coding: utf-8

# In[1]:


# mount on google drive
from google.colab import drive
drive.mount('/content/drive/')
# go to your code files directory
import os
os.chdir("/content/drive/My Drive/Sar_WaterExt_Code")
# !ls

# In[2]:


try:
    get_ipython().magic(u'tensorflow_version 2.x')
except Exception:
    pass
import tensorflow as tf
import numpy as np
from utils.utils import readTiff, imgPatch, imsShow
from dataPrepare.tfrecord_s1 import tfrecord_s1_scene, tfrecord_s1_patch
import pprint as pp


# In[4]:


import os
os.chdir("/content/drive/My Drive/Sar_WaterExt_Data")
## data read path
s1_ascend = '/content/drive/My Drive/Sar_WaterExt_Data/s1_ascend/*.tif'
s1_descend = '/content/drive/My Drive/Sar_WaterExt_Data/s1_descend/*.tif'
watTruth = '/content/drive/My Drive/Sar_WaterExt_Data/s2_img_truth/*_wat_truth.tif'
## data write path
traScene_file = '/content/drive/My Drive/Sar_WaterExt_Data/traScene.tfrecords'
# traPatch_file = '/content/drive/My Drive/Sar_WaterExt_Data/traPatch_256_512_2048_100x.tfrecords'
evaScene_file = '/content/drive/My Drive/Sar_WaterExt_Data/evaScene.tfrecords'
evaPatch_file = '/content/drive/My Drive/Sar_WaterExt_Data/evaPatch_256_512_2048_100x.tfrecords'

s1_ascend_path = sorted(tf.io.gfile.glob(s1_ascend))
s1_descend_path = sorted(tf.io.gfile.glob(s1_descend))
watTruth_path = sorted(tf.io.gfile.glob(watTruth))

traScene_path = list(zip(s1_ascend_path, s1_descend_path, watTruth_path))[0:-5]
print(len(traScene_path))
evaScene_path = list(zip(s1_ascend_path, s1_descend_path, watTruth_path))[-5:]
print(len(evaScene_path))


# In[ ]:


tfrecord_traScene_ins = tfrecord_s1_scene()

## 15/20(01-15) scenes for training
### Write to a `.tfrecords` file for model training.
with tf.io.TFRecordWriter(traScene_file) as writer:    
    for s1_ascend_path, s1_descend_path, truth_water_path in traScene_path:
        print(s1_ascend_path)
        s1_ascend, _, _, _, _, _ = readTiff(s1_ascend_path)
        s1_descend, _, _, _, _, _ = readTiff(s1_descend_path)
        truth, _, _, _, _, _ = readTiff(truth_water_path)
        ### normalization
        s1_ascend_nor = (s1_ascend-np.nanmin(s1_ascend))/(np.nanmax(s1_ascend)-np.nanmin(s1_ascend))
        s1_descend_nor = (s1_descend-np.nanmin(s1_descend))/(np.nanmax(s1_descend)-np.nanmin(s1_descend))
        ### tfrecord writing
        imsShow([s1_ascend_nor, s1_descend_nor, truth],['s1_ascend', 's1_descend', 's1_truth'],\
                                            [2,2,0], [(0,1,0),(0,1,0),(0,0,0)], figsize=(10,3))
        tf_example = tfrecord_traScene_ins.scene_example_serilize(s1_ascend_nor, \
                                            s1_descend_nor, truth)
        writer.write(tf_example.SerializeToString())


# In[ ]:


# tfrecord_evaScene_ins = tfrecord_s1_scene()

# ### Write scenes to a `.tfrecords` file.
# ## 5/20(16,17,18,19,20) scenes for evaluation
# with tf.io.TFRecordWriter(evaScene_file) as writer:  
#     for s1_ascend_path, s1_descend_path, truth_water_path in evaScene_path:
#         print(s1_ascend_path)
#         s1_ascend, _, _, _, _, _ = readTiff(s1_ascend_path)
#         s1_descend, _, _, _, _, _ = readTiff(s1_descend_path)
#         truth, _, _, _, _, _ = readTiff(truth_water_path)
#         s1_ascend_nor = (s1_ascend-np.nanmin(s1_ascend))/(np.nanmax(s1_ascend)-np.nanmin(s1_ascend))
#         s1_descend_nor = (s1_descend-np.nanmin(s1_descend))/(np.nanmax(s1_descend)-np.nanmin(s1_descend))
#         imsShow([s1_ascend_nor, s1_descend_nor, truth],['s1_ascend', 's1_descend','s1_truth'], \
#                                                 [2,2,0], [(0,1,0),(0,1,0),(0,0,0)], figsize=(10,3))
#         tf_example = tfrecord_evaScene_ins.scene_example_serilize(s1_ascend_nor, \
#                                             s1_descend_nor, truth)
#         writer.write(tf_example.SerializeToString())


# In[ ]:


# ## write the training patches to a tfrecord file from scenes data
# ##########################################
# ### 1. load the training scenes and convert it to patches group. 
# tfrecord_scene_ins = tfrecord_s1_scene(scale_low = 256, scale_mid = 512, scale_high = 2048)
# traScene = tf.data.TFRecordDataset(traScene_file)
# traPatch = traScene.map(tfrecord_scene_ins.parse_sceneBand).map(tfrecord_scene_ins.parse_sceneShape)\
                                            # .map(tfrecord_scene_ins.toPatchGroup_fromScene)
# print(traPatch)

# ### 2. write the patches to a .tfrecord file.
# tfrecord_patch_ins = tfrecord_s1_patch()
# with tf.io.TFRecordWriter(traPatch_file) as writer:
#     for i in range(100):
#         print(i)
#         for img_high, img_mid, img_low, truth_low in traPatch:
#             imsShow([img_high, img_mid, img_low, truth_low], ['img_high', 'img_mid', 'img_low', 'truth_low'], \
#                                                 [2,2,2,0], [(2,1,0),(2,1,0),(2,1,0), (0,0,0)], figsize=(12,3))
#             tf_example = tfrecord_patch_ins.patch_example_serilize(img_local.numpy(), truth_local.numpy(), img_global.numpy())
#             writer.write(tf_example.SerializeToString())


# In[6]:


## write the evaluation patches to a tfrecord file from image scene data
##########################################
### 1. load the evaluation scenes and convert it to patches group.
tfrecord_scene_ins = tfrecord_s1_scene(scale_low = 256, scale_mid = 512, scale_high = 2048)
evaScene = tf.data.TFRecordDataset(evaScene_file)
evaPatch = evaScene.map(tfrecord_scene_ins.parse_sceneSample, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
            .map(tfrecord_scene_ins.parse_sceneShape, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
            .map(tfrecord_scene_ins.toPatchGroup_fromScene, num_parallel_calls=tf.data.experimental.AUTOTUNE)
print(evaPatch)

### 2. write the patches to a .tfrecord file.
tfrecord_patch_ins = tfrecord_s1_patch()
with tf.io.TFRecordWriter(evaPatch_file) as writer:
    for i in range(100):
        print(i)
        for (img_high, img_mid, img_low), truth_low in evaPatch:
            # imsShow([img_high, img_mid, img_low, truth_low],['img_high', 'img_mid', 'img_low', 'truth_low'], \
            #                                 [2,2,2,0], [(2,1,0),(2,1,0),(2,1,0), (0,0,0)], figsize=(12,3))
            tf_example = tfrecord_patch_ins.patch_example_serilize(img_high.numpy(),\
                                            img_mid.numpy(),img_low.numpy(), truth_low.numpy())
            writer.write(tf_example.SerializeToString())

