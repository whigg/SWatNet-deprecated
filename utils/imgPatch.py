import numpy as np

class imgPatch():
    '''
    description: 1. remote sensing image to patches
                 2. patches to remote sensing image 
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

