import tensorflow as tf

def h_sigmoid(x):
    return tf.nn.relu6(x + 3) / 6
