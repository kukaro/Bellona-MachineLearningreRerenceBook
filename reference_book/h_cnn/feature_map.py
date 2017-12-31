import tensorflow as tf

input_batch = tf.constant([
    [  # first input
        [[0.0], [1.0]],
        [[2.0], [3.0]]
    ],
    [  # second input
        [[2.0], [4.0]],
        [[6.0], [8.0]]
    ]
])

kernel = tf.constant([
    [
        [[1.0, 2.0]]
    ]
])

strides = [1, 1, 1, 1]


def run():
    sess = tf.Session()
    conv2d = tf.nn.conv2d(input_batch, kernel, strides=strides, padding="SAME")

    lower_right_image_pixel = sess.run(input_batch)[0][1][1]
    lower_right_kernel_pixel = sess.run(conv2d)[0][1][1]
    print(lower_right_image_pixel)
    print(lower_right_kernel_pixel)
