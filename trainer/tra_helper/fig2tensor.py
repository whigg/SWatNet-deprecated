import tensorflow as tf
import numpy as np
from PIL import Image

def fig2tensor(figure):
    """
    Converts the matplotlib plot specified by 'figure' to tf.Tensor data.
    """
    figure.canvas.draw()
    w,h = figure.canvas.get_width_height()
    buf = np.frombuffer(figure.canvas.tostring_rgb(), dtype=np.uint8)
    buf.shape=(w,h,3)
    buf = np.roll(buf,3,axis=2)
    img = Image.frombytes("RGB",(w,h),buf.tostring())
    img_tf = tf.convert_to_tensor(np.array(img))
    return tf.expand_dims(img_tf, 0)
