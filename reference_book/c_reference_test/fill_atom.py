import tensorflow as tf


def run():
    a = tf.zeros([6, 7])
    b = tf.ones([5, 5])
    c = tf.random_uniform([3, 3, 3], minval=0, maxval=10)
    d = tf.random_normal([4, 4, 4], mean=0.0, stddev=2.0)
    sess = tf.Session()
    print(sess.run(a))
    print(sess.run(b))
    print(sess.run(c))
    print(sess.run(d))
