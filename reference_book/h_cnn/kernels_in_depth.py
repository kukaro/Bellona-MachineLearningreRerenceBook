import tensorflow as tf
import os

image_filename = 'image.jpg'
filename_queue = tf.train.string_input_producer(
    tf.train.match_filenames_once([os.path.dirname(__file__) + '/' + image_filename]))
image_reader = tf.WholeFileReader()
_,image_file = image_reader.read(filename_queue)
image = tf.image.decode_jpeg(image_file)

kernel = tf.constant([
    [
        [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
        [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
        [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]
    ],
    [
        [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
        [[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 8.0]],
        [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]
    ],
    [
        [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
        [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
        [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]
    ]
])
with tf.Session() as sess:
    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    print(sess.run(image_file))
    coord.request_stop()
    coord.join(threads)
    sess.run(image)
    conv2d = tf.nn.conv2d(image, kernel, strides=[1, 1, 1, 1], padding='SAME')
# activation_map = sess.run(tf.minimum(tf.nn.relu(conv2d), 255))
