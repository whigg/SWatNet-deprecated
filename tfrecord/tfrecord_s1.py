## author: xin luo; date: 2021.3.15

import tensorflow as tf

def int64_feature(value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def float_feature(value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

class tfrecord_s1_scene():
    '''
    write and parse the tfrecord file in form of image scene.
    '''
    def __init__(self, scale_low=256, scale_mid =512, scale_high=2048):
        "the augements should be determined in the tfrecord file parsing stage"
        self.feaBand_name = ['s1_ascend_vv','s1_ascend_vh','s1_decend_vv','s1_decend_vh']
        self.truBand_name = ['water_truth']
        self.mergeBand_name = self.feaBand_name+self.truBand_name
        self.fea_num = len(self.feaBand_name)
        self.patch_size = scale_low
        self.scale_low = scale_low
        self.scale_mid = scale_mid
        self.scale_high = scale_high
        ### feature dictionary for parsing the serialized tfrecord file.        
        self.scene_fea_description = {
            'sceneBand_shape': tf.io.FixedLenFeature(shape=[2,], dtype=tf.int64),
            's1_ascend_vv': tf.io.VarLenFeature(dtype=tf.float32),
            's1_ascend_vh': tf.io.VarLenFeature(dtype=tf.float32),
            's1_decend_vv': tf.io.VarLenFeature(dtype=tf.float32),
            's1_decend_vh': tf.io.VarLenFeature(dtype=tf.float32),
            'water_truth': tf.io.VarLenFeature(dtype=tf.float32)
                }        
    ################################################################
    ## write the tfrecord file to image scene
    def scene_example_serilize(self, s1_ascend, s1_descend, truth):
        '''
        write and parse the tfrecord file which stored the image patch data.
        ''' 
        feaDict_series = {
            'sceneBand_shape': int64_feature(s1_ascend[:,:,0].shape),
            's1_ascend_vv': float_feature(s1_ascend[:,:,0].flatten()),
            's1_ascend_vh': float_feature(s1_ascend[:,:,1].flatten()),
            's1_decend_vv': float_feature(s1_descend[:,:,0].flatten()),
            's1_decend_vh': float_feature(s1_descend[:,:,1].flatten()),
            'water_truth': float_feature(truth.flatten()),
                }
        return tf.train.Example(features=tf.train.Features(feature=feaDict_series))

    ##############################################
    ### parse the tfrecord files to image scenes. 
    def parse_sceneSample(self, example_proto):
        '''Parse the input tf.train.Example proto using the dictionary above'''
        return tf.io.parse_single_example(example_proto, self.scene_fea_description)
    
    def parse_sceneShape(self, example_parsed):
        '''parse the tfrecord file to the original shape'''
        for fea in self.mergeBand_name:
            example_parsed[fea] = tf.sparse.to_dense(example_parsed[fea])
            example_parsed[fea] = tf.reshape(example_parsed[fea], example_parsed['sceneBand_shape'])
        return example_parsed
    
    def toScenePair(self, inputs):
        '''
        inputs: the parsed image scene data.
        outputs: 1) feature bands of the image scene, 2) truth band of the image scene.
        '''
        inputsList = [inputs.get(key) for key in self.mergeBand_name]
        stacked = tf.stack(inputsList, axis=2)
        return stacked[:,:,:self.fea_num], stacked[:,:,self.fea_num:]
    
    #####################################################
    ## build mainly for the multi-scale training patches generation
    def toPatchPair_fromScene(self, inputs):
        '''
        inputs: the parsed image scene data.
        outputs: 1) image patch of the feature bands, 2) image patch of the truth band.
        '''
        inputsList = [inputs.get(key) for key in self.mergeBand_name]
        stacked = tf.stack(inputsList, axis=2)        
        cropped_stacked = tf.image.random_crop(
                        stacked, size=[self.patch_size, self.patch_size, len(self.mergeBand_name)])
        return cropped_stacked[:,:,:self.fea_num], cropped_stacked[:,:,self.fea_num:]

    @tf.function
    def toPatchGroup_fromScene(self, inputs):
        '''
        inputs:  the tfrecord features in tfrecord.
        outputs: 1) the down sampled high-scale image; 2) the down sampled middle-scale image, 
                3) the low-scale image, and 4) the truth image corresponding to low-scale image
        '''
        crop_start_low = (self.scale_high-self.scale_low)//2 
        crop_start_low = (self.scale_high-self.scale_low)//2
        crop_start_mid = (self.scale_high-self.scale_mid)//2 
        crop_start_mid = (self.scale_high-self.scale_mid)//2 
        inputsList = [inputs.get(key) for key in self.mergeBand_name]
        stacked_image = tf.stack(inputsList, axis=2)        
        image_high = tf.image.random_crop(stacked_image, \
                                    size=[self.scale_high, self.scale_high, len(inputsList)])        
        image_mid = tf.image.crop_to_bounding_box(image_high, \
                                    offset_height=crop_start_mid, offset_width=crop_start_mid,\
                                    target_height=self.scale_mid, target_width=self.scale_mid)
        image_low = tf.image.crop_to_bounding_box(image_high, \
                                    offset_height=crop_start_low, offset_width=crop_start_low,\
                                    target_height=self.scale_low, target_width=self.scale_low)        
        image_high_down = tf.image.resize(image_high, [self.patch_size, self.patch_size], method='area')
        image_mid_down = tf.image.resize(image_mid, [self.patch_size, self.patch_size], method='area')
        return (image_high_down[:,:,:self.fea_num], image_mid_down[:,:,:self.fea_num],\
                                        image_low[:,:,:self.fea_num]), image_low[:,:,self.fea_num:]

class tfrecord_s1_patch():
    '''
    write and parse the tfrecord file in form of image patch.
    ** build for the evaluation data writing and parsing.
    '''
    def __init__(self, patch_size = 256):
        "the augements should be determined in the tfrecord files parsing stage"
        self.patch_size = patch_size
        self.groupPatch_name = ['img_high', 'img_mid', 'img_low','truth_low']
        ### feature dictionary for parsing the serialized tfrecord file.
        self.patch_fea_description = {
            'img_high_shape': tf.io.FixedLenFeature(shape=[3,], dtype=tf.int64),
            'img_mid_shape': tf.io.FixedLenFeature(shape=[3,], dtype=tf.int64),
            'img_low_shape': tf.io.FixedLenFeature(shape=[3,], dtype=tf.int64),
            'truth_low_shape': tf.io.FixedLenFeature(shape=[3,], dtype=tf.int64),
            'img_high': tf.io.FixedLenFeature(shape=[self.patch_size*self.patch_size*4,],dtype=tf.float32),
            'img_mid': tf.io.FixedLenFeature(shape=[self.patch_size*self.patch_size*4,],dtype=tf.float32),
            'img_low': tf.io.FixedLenFeature(shape=[self.patch_size*self.patch_size*4,],dtype=tf.float32),
            'truth_low': tf.io.FixedLenFeature(shape=[self.patch_size*self.patch_size*1,],dtype=tf.float32)
            }

    ################################################################
    ##  write to the tfrecord file to image patch
    def patch_example_serilize(self, img_high, img_mid, img_low, truth_low):
        '''
        feature dictionary used to serialize the evaluation data into tfrecord file
        '''
        feaDict_series = {
            'img_high_shape': int64_feature(img_high.shape),
            'img_mid_shape': int64_feature(img_mid.shape),
            'img_low_shape': int64_feature(img_low.shape),
            'truth_low_shape': int64_feature(truth_low.shape),
            'img_high': float_feature(img_high.flatten()),
            'img_mid': float_feature(img_mid.flatten()),
            'img_low': float_feature(img_low.flatten()),
            'truth_low': float_feature(truth_low.flatten()),
        }
        return tf.train.Example(features=tf.train.Features(feature=feaDict_series))

    ##############################################
    ### parse the tfrecord files to image patch. 
    def parse_patchSample(self, example_proto):
        '''Parse the input `tf.train.Example` proto using the dictionary above.'''  
        return tf.io.parse_single_example(example_proto, self.patch_fea_description)

    def parse_patchShape(self, example_parsed):
        for fea in self.groupPatch_name:
            ## bug: if reshaping by using example_parsed[shape_key], the obtained shape is None.
            # shape_key = fea + '_shape'
            # example_parsed[fea] = tf.reshape(example_parsed[fea], example_parsed[shape_key])
            if fea == 'truth_low':
                example_parsed[fea] = tf.reshape(example_parsed[fea], [256,256,1])                
            else:
                example_parsed[fea] = tf.reshape(example_parsed[fea], [256,256,4])
        return example_parsed

    def toPatchGroup(self, inputs):
        return (inputs.get(self.groupPatch_name[0]), inputs.get(self.groupPatch_name[1]), \
                                inputs.get(self.groupPatch_name[2])), inputs.get(self.groupPatch_name[3])

