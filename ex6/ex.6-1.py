# 여러입력값을 갖는 다중로지스틱 회귀
import tensorflow as tf
import numpy as np
from numpy.core._multiarray_umath import dtype

x_data = np.array([[2, 3], [4, 3], [6, 4], [8, 6], [10, 7], [12, 8], [14, 9]])
y_data = np.array([0, 0, 0, 1, 1, 1, 1]).reshape(7, 1)  # 행렬을 변경 7행 1열
print("x_data: {}".format(x_data))
print("y_data: {}".format(y_data))

# 입력값을 플레이스 홀더에 저장
# placeholder('데이터형", "행렬의차원", "이름")
X = tf.placeholder(tf.float64, shape=[None, 2]) # 2차원 행렬
Y = tf.placeholder(tf.float64, shape=[None, 1]) # 1차원 행렬
print("X: {}".format(X))
print("Y: {}".format(Y))

# 기울기a와 바이어스b의 값을 랜덤으로 정함
# [2, 1]: 2개의 값이 들어오고 나가는값은 1개
a = tf.Variable(tf.random_uniform([2, 1], dtype=tf.float64))
b = tf.Variable(tf.random_uniform([1], dtype=tf.float64))
print("a: {}".format(a))
print("b: {}".format(b))

# 시그모이드함수 단일일 경우: y = 1 / (1 + e^(-ax + b))
# 다중일 경우 a1x1 + a2x2로 변경됨 [a1, a2] * [x1, x2]
# ([a1, a2] * [x1, x2] + b)
# matmul로 행렬의 곱을 해줌
y = tf.sigmoid(tf.matmul(X, a) + b)

# 로그함수
# 오차 = -평균(y * log h + (1 - y) * log(1 - h))
loss = -tf.reduce_mean(y * tf.log(y) + (1 - y) * tf.log(1 - y))

# 학습률 값
learning_rate = 0.1

# 경사하강법으로 오차를 최소로하는 값을 찾음
gradient_decent = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

predicted = tf.cast(y > 0.4, dtype=tf.float64)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float64))
print("predicted: {}".format(predicted))
print("accuracy: {}".format(accuracy))


# [공부한시간, 과외횟수]
new_x = np.array([7, 6]).reshape(1, 2)
print("new_x : {}".format(new_x))

# 학습
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    new_y = sess.run(y, feed_dict={X: new_x})
    print("공부한 시간: %d, 과외수업횟수: %d" % (new_x[:, 0], new_x[:, 1]))
    print("합격 가능성: %.2f %%" % (new_y * 100))

    for i in range(3001):
        a_, b_, loss_, _ = sess.run([a, b, loss, gradient_decent], feed_dict={X: x_data, Y: y_data})
        if (i + 1) % 300 == 0:
            print("step=%d, a1=%.4f, a2=%.4f, b=%.4f, loss=%.4f" % (i + 1, a_[0], a_[1], b_, loss_))
