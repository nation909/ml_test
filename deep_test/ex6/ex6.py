# 로지스틱회귀

import tensorflow as tf
import numpy as np

data = [[2, 0], [4, 0], [6, 0], [8, 1], [10, 1], [12, 1], [14, 1]]

# x_data: 공부한시간 y: 과외받은횟수
x_data = [x_row[0] for x_row in data]
y_data = [y_row[1] for y_row in data]
print("x_data: {}".format(x_data))
print("y_data: {}".format(y_data))

# a: 그래프의 경사도 b: 그래프의 좌/우 이동
# a, b의 값을 랜덤으로 뽑음
a = tf.Variable(tf.random_normal([1], dtype=tf.float64, seed=0))
b = tf.Variable(tf.random_normal([1], dtype=tf.float64, seed=0))

# 시그모이드함수
# y = 1 / (1 + e^(-ax + b))
y = 1 / (1 + np.e ** (-a * x_data + b))
print("y: {} type: {}".format(y, type(y)))

# 로그함수
# 오차 = -평균(y * log h + (1 - y) * log(1 - h))
loss = -tf.reduce_mean(np.array(y_data) * tf.log(y) + (1 - np.array(y_data)) * tf.log(1 - y))
print("loss: {}".format(loss))

# 학습률 값
learning_rate = 0.5

# 경사하강법으로 오차를 최소로하는 값을 찾음
gradient_decent = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# 결과값 출력
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    X = tf.placeholder(tf.float64, shape=[None, 1])  # 1차원 행렬
    result = sess.run(y, feed_dict={X: x_data})
    print("result: {}".format(result))

    for i in range(60001):
        sess.run(gradient_decent)
        if i % 6000 == 0:
            print("Epoch: %.f, loss = %.4f, 기울기a = %.4f, y절편 = %.4f" % (i, sess.run(loss), sess.run(a), sess.run(b)))
