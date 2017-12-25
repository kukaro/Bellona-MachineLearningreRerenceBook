import tensorflow as tf
import matplotlib.pyplot as plt

graph = tf.get_default_graph()
# saver = tf.train.Saver()

W = tf.Variable(tf.zeros([2, 1]), name='weights')
b = tf.Variable(0., name='bias')


def inference(X):
    return tf.matmul(X, W) + b


def loss(X, Y):
    Y_predicted = inference(X)
    return tf.reduce_sum(tf.squared_difference(Y, Y_predicted))


def inputs():
    weight_age = [[84, 46], [73, 20], [65, 52], [70, 30], [76, 57], [69, 25],
                  [63, 28], [72, 36], [79, 57], [75, 44], [27, 24], [89, 31],
                  [65, 52], [57, 23], [59, 60], [69, 48], [60, 34], [79, 51],
                  [75, 50], [82, 34], [59, 46], [67, 23], [85, 37], [55, 40],
                  [63, 30]]
    blood_fat_content = [354, 190, 405, 263, 451, 302, 288, 385, 402, 365,
                         209, 290, 346, 254, 395, 434, 220, 374, 308, 220,
                         311, 181, 274, 303, 244]
    return tf.to_float(weight_age), tf.to_float(blood_fat_content), weight_age, blood_fat_content


def train(total_loss):
    learning_rate = 0.0000001
    return tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)


def evaluate(sess, X, Y):
    print(sess.run(inference([[89., 25.]])))
    print(sess.run(inference([[65., 25.]])))

def run():
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        X, Y, weight_age, blood_fat_content = inputs()
        total_loss = loss(X, Y)
        train_op = train(total_loss)
        coord = tf.train.Coordinator()

        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        training_steps = 1000
        weight = [weight_age[x][0] for x in range(25)]
        plt.plot(weight, blood_fat_content, 'ro')

        for step in range(training_steps):
            sess.run([train_op])

            if step % 10 == 0:
                print("loss: ", sess.run(total_loss))
        wp, bp = sess.run([W, b])
        evaluate(sess, X, Y)

        #start addition code
        weight_age_f = [[float(weight_age[x][0]), float(weight_age[x][1])] for x in range(25)]
        yp = [sess.run(inference([weight_age_f[x]])) for x in range(25)]
        ypp = [yp[x][0][0] for x in range(25)]
        print(ypp)
        plt.plot(weight, ypp, 'bo')
        plt.show()
        #and addition code

        coord.request_stop()
        coord.join(threads)
        sess.close()

'''
https://tensorflow.blog/%EA%B2%BD%EC%82%AC%ED%95%98%EA%B0%95%EB%B2%95-tf-gradients/
참고 사이트

선형회귀(기울기 하강)는 제곱에러(squared error:L2)를 사용하여 손실함수를 만든다.

pyplot은 무게를 기준으로 출력했으며 빨간색은 원래값으고 파란색은 우리가 만든 함수에 값을 대입한 결과이다.
실제 선형이 아닌 이유는 실제로 들어가는 벡터가 1차원(무게)만 있는 것이 아니라 2차원(무게,나이)이기 때문이다.
'''
