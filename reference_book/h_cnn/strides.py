import tensorflow as tf

# 0.0부터 5.5까지를 0.1간격으로 출력
input_batch = tf.constant([  # first input (6*6*1)
    [[[float(x + y / 10)] for x in range(0, 6)] for y in range(0, 6)]
])

kernel = tf.constant([  # kernel (3*3*1)
    [[[0.0]], [[0.5]], [[0.0]]],
    [[[0.0]], [[1.0]], [[0.0]]],
    [[[0.0]], [[0.5]], [[0.0]]]
])

# 크기를 3*3*1 단위로 잘라서 시행
strides = [1, 3, 3, 1]

conv2d = tf.nn.conv2d(input_batch, kernel, strides=strides, padding='SAME')

sess = tf.Session()
print(sess.run(conv2d))
# print(sess.run(input_batch))
