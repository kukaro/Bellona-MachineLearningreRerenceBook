import tensorflow as tf


def run():
    a = tf.Variable(0)
    init = tf.initialize_all_variables()

    sess1 = tf.Session()
    sess2 = tf.Session()

    sess1.run(init)
    sess2.run(init)

    print(sess1.run(a.assign_add(5)))
    print(sess2.run(a.assign_add(7)))
    print(sess1.run(a.assign_add(5)))
    print(sess2.run(a.assign_add(7)))

    '''
    변수 별로 관리되는게 아니라 세션별로 관리된다.
    '''
