import tensorflow as tf


def run():
    a = tf.zeros([6, 7])
    b = tf.ones([5, 5])
    c = tf.random_uniform([3, 3, 3], minval=0, maxval=10)
    d = tf.random_normal([4, 4, 4], mean=0.0, stddev=2.0)
    e = tf.truncated_normal([2, 2, 2], mean=0, stddev=2.0)
    sess = tf.Session()
    print(sess.run(a))
    print(sess.run(b))
    print(sess.run(c))
    print(sess.run(d))
    print(sess.run(e))

    '''
    truncated_normal은 평균에서 표준편차가 2배이상 나는 것은 생성하지 않는다.
    '''
