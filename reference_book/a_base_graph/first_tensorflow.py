import tensorflow as tf


def run():
    a = tf.constant(5, name='input_a')
    b = tf.constant(3, name='input_b')
    c = tf.multiply(a, b, name='mul_c')
    d = tf.add(a, b, name='add_d')
    e = tf.add(c, d, name='add_e')
    sess = tf.Session()
    print(sess.run(e))
