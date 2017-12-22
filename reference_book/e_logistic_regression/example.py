import tensorflow as tf

W = tf.Variable(tf.zeros([5, 1]), name='weights')
b = tf.Variable(0., name='bias')


def combine_inputs(X):
    return tf.matmul(X, W) + b


def inference(X):
    return tf.sigmoid(combine_inputs(X))
