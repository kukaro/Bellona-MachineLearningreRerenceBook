import tensorflow as tf

image_batch = tf.constant([
    [  # First Image
        [[0, 255, 0], [0, 255, 0], [0, 255, 0]],
        [[0, 255, 0], [0, 255, 0], [0, 255, 0]]
    ],
    [  # Second Image
        [[0, 0, 255], [0, 0, 255], [0, 0, 255]],
        [[0, 0, 255], [0, 0, 255], [0, 0, 255]]
    ]
])


def run():
    image_batch.get_shape()
    sess = tf.Session()
    print(sess.run(image_batch))
    print(image_batch.get_shape())
