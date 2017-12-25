import tensorflow as tf
import os

W = tf.Variable(tf.zeros([4, 3]), name='weights')
b = tf.Variable(tf.zeros([3]), name='bias')


def combine_inputs(X):
    return tf.matmul(X, W) + b


def inference(X):
    return tf.nn.softmax(combine_inputs(X))


def loss(X, Y):
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=combine_inputs(X), labels=Y))


def train(total_loss):
    learning_rate = 0.01
    return tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)


def evaluate(sess, X, Y):
    # predicted : 예상되는
    predicted = tf.cast(tf.arg_max(inference(X), 1), tf.int32)
    print(sess.run(tf.reduce_mean(tf.cast(tf.equal(predicted, Y), tf.float32))))


def inputs():
    sepal_length, sepal_width, petal_length, petal_width, label = \
        read_csv(100, 'iris.data', [[0.0], [0.0], [0.0], [0.0], ['']])

    label_number = tf.to_int32(tf.arg_max(tf.to_int32(tf.stack(
        [tf.equal(label, ['Iris-setosa']), tf.equal(label, ['Iris-versicolor']), tf.equal(label, ['Iris-virginica'])])),
        0))

    features = tf.transpose(tf.stack([sepal_length, sepal_width, petal_length, petal_width]))
    return features, label_number


def read_csv(batch_size, file_name, record_defaults):
    filename_queue = tf.train.string_input_producer([os.path.dirname(__file__) + '/' + file_name])
    reader = tf.TextLineReader(skip_header_lines=1)
    key, value = reader.read(filename_queue)

    '''
    tensorflow.python.framework.errors_impl.InvalidArgumentError
    해보니까 디코드할때 디코드할 대상의 파일에 제일아래에 개행이 두개이상 있으면 저에러가 발생함.
    '''
    decoded = tf.decode_csv(value, record_defaults=record_defaults)

    return tf.train.shuffle_batch(decoded, batch_size=batch_size, capacity=batch_size * 50,
                                  min_after_dequeue=batch_size)


def run():
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        X, Y = inputs()
        total_loss = loss(X, Y)
        train_op = train(total_loss)
        coord = tf.train.Coordinator()

        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        training_steps = 1000
        print(sess.run(X))
        for step in range(training_steps):
            sess.run([train_op])

            if step % 10 == 0:
                print("loss: ", sess.run(total_loss))
        wp, bp = sess.run([W, b])
        evaluate(sess, X, Y)

        coord.request_stop()
        coord.join(threads)
        sess.close()
