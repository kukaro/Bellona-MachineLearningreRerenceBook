import tensorflow as tf


def run():
    a = tf.constant(3, name='a')
    b = tf.constant([[10,20],[30,40]],name='b')
    c = tf.add(a,b)
    sess = tf.Session()
    print(sess.run(c,feed_dict={a:10}))
