import config
import sys
sys.path.append(config.root_dir)
from tfrecord.tfrecord_s1 import tfrecord_s1_scene, tfrecord_s1_patch
from tra_helper.image_aug import image_aug
import tensorflow as tf

def get_tra_dset():
  ''' load training scenes from tfrecord file'''
  tfrecord_scene_ins = tfrecord_s1_scene()
  traScene = tf.data.TFRecordDataset(config.tra_scene_file)
  tra_patch_group = traScene.map(tfrecord_scene_ins.parse_sceneSample)\
                              .map(tfrecord_scene_ins.parse_sceneShape)\
                              .cache()\
                              .map(tfrecord_scene_ins.toPatchGroup_fromScene,\
                                num_parallel_calls=tf.data.experimental.AUTOTUNE)\
                              .map(image_aug, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
                              .repeat(4)\
                              .prefetch(tf.data.experimental.AUTOTUNE)

  # tra_dset = tra_patch_group.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
  tra_dset = tra_patch_group.batch(config.BATCH_SIZE)
  return tra_dset

def get_eva_dset():
    '''load the evaluation patches data.'''
    tfrecord_patch_ins = tfrecord_s1_patch()
    test_dset = tf.data.TFRecordDataset(config.eva_patch_file)
    test_dset = test_dset.map(tfrecord_patch_ins.parse_patchSample)\
                        .map(tfrecord_patch_ins.parse_patchShape)\
                        .map(tfrecord_patch_ins.toPatchGroup)\
                        .cache()

    test_dset = test_dset.batch(config.BATCH_SIZE)
    return test_dset

