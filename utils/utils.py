
import tensorflow as tf
from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt
import random

### tiff image reading
def readTiff(path_in):
    '''
    return: numpy array, dimentions order: (row, col, band)
    '''
    RS_Data=gdal.Open(path_in)
    im_col = RS_Data.RasterXSize  # 
    im_row = RS_Data.RasterYSize  # 
    im_bands =RS_Data.RasterCount  # 
    im_geotrans = RS_Data.GetGeoTransform()  # 
    im_proj = RS_Data.GetProjection()  # 
    RS_Data = RS_Data.ReadAsArray(0, 0, im_col, im_row)  # 
    if im_bands > 1:
        RS_Data = np.transpose(RS_Data, (1, 2, 0)).astype(np.float)  # 
        return RS_Data, im_geotrans, im_proj, im_row, im_col, im_bands
    else:
        return RS_Data,im_geotrans,im_proj,im_row,im_col,im_bands

###  .tiff image write
def writeTiff(im_data, im_geotrans, im_proj, path_out):
    '''
    im_data: tow dimentions (order: row, col),or three dimentions (order: row, col, band)
    '''
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    if len(im_data.shape) == 3:
        im_data = np.transpose(im_data, (2, 0, 1))
        im_bands, im_height, im_width = im_data.shape
    else:
        im_bands,(im_height, im_width) = 1,im_data.shape
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path_out, im_width, im_height, im_bands, datatype)
    if(dataset!= None):
        dataset.SetGeoTransform(im_geotrans)    # 
        dataset.SetProjection(im_proj)      # 
    if im_bands > 1:
        for i in range(im_bands):
            dataset.GetRasterBand(i+1).WriteArray(im_data[i])
        del dataset
    else:
        dataset.GetRasterBand(1).WriteArray(im_data)
        del dataset

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

def imgShow(img, col_bands=(2,1,0), clip_percent=2, per_band_clip='False'):
    '''
    Arguments:
        img: (row, col, band) or (row, col)
        num_bands: a list/tuple, [red_band,green_band,blue_band]
        clip_percent: for linear strech, value within the range of 0-100. 
        per_band: if 'True', the band values will be clipped by each band respectively. 
    '''
    img = img/(np.amax(img)+0.00001)
    img = np.squeeze(img)
    if np.isnan(np.sum(img)) == True:
        where_are_NaNs = np.isnan(img)
        img[where_are_NaNs] = 0
    elif np.min(img) == np.max(img):
        if len(img.shape) == 2:
            plt.imshow(np.clip(img, 0, 1),vmin=0,vmax=1) 
        else:
            plt.imshow(np.clip(img[:,:,0], 0, 1),vmin=0,vmax=1)
    else:
        if len(img.shape) == 2:
            img_color = img
        else:
            img_color = img[:,:,[col_bands[0], col_bands[1], col_bands[2]]]    
        img_color_clip = np.zeros_like(img_color)
        if per_band_clip == 'True':
            for i in range(3):
                img_color_hist = np.percentile(img_color[:,:,i], [clip_percent, 100-clip_percent])
                img_color_clip[:,:,i] = (img_color[:,:,i]-img_color_hist[0])\
                                    /(img_color_hist[1]-img_color_hist[0])
        else:
            img_color_hist = np.percentile(img_color, [clip_percent, 100-clip_percent])
            img_color_clip = (img_color-img_color_hist[0])\
                                    /(img_color_hist[1]-img_color_hist[0])
        plt.imshow(np.clip(img_color_clip, 0, 1),vmin=0,vmax=1)

def imsShow(img_list, imgName_list, clip_list, 
                            col_bands_list, figsize = (12,12)):
    figure = plt.figure(figsize=figsize)
    for i in range(len(img_list)):
        plt.subplot(1, len(img_list), i+1)
        plt.title(imgName_list[i])
        imgShow(img_list[i],
            col_bands=col_bands_list[i], clip_percent=clip_list[i])
        plt.axis('off')
    # plt.show()
    return figure

class imgPatch():
    '''
    remote sensing image to patches
    patches to remote sensing image 
    '''
    def __init__(self, img, patch_size, overlay):
        self.patch_size = patch_size
        self.overlay = overlay
        self.img = img[:,:,np.newaxis] if len(img.shape) == 2 else img
        self.img_row = img.shape[0]
        self.img_col = img.shape[1]

    def toPatch(self):
        patch_list = []
        img_expand = np.pad(self.img, ((self.overlay//2, self.patch_size), 
                                          (self.overlay//2, self.patch_size), (0,0)), 'constant')
        patch_step = self.patch_size - self.overlay
        img_patch_row = (img_expand.shape[0]-self.overlay)//patch_step
        img_patch_col = (img_expand.shape[1]-self.overlay)//patch_step
        for i in range(img_patch_row):
            for j in range(img_patch_col):
                patch_list.append(img_expand[i*patch_step:i*patch_step+self.patch_size,
                                                j*patch_step:j*patch_step+self.patch_size, :])
        return patch_list, img_patch_row, img_patch_col

    def toImage(self, patch_list, img_patch_row, img_patch_col):
        patch_list = [patch[self.overlay//2:-self.overlay//2, self.overlay//2:-self.overlay//2,:]
                                                        for patch in patch_list]
        patch_list = [np.hstack((patch_list[i*img_patch_col:i*img_patch_col+img_patch_col]))
                                                        for i in range(img_patch_row)]
        img_array = np.vstack(patch_list)
        img_array = img_array[0:self.img_row, 0:self.img_col, :]
        return img_array