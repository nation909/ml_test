import tensorflow as tf

# y = a1 * x1 + a2 * x2 + b

# 공부한시간(x1), 과외수업횟수(x2), 실제성적(y)
data = [[2, 0, 81], [4, 4, 93], [6, 2, 91], [8, 3, 97]]
x1 = [x_row1[0] for x_row1 in data]
x2 = [x_row2[1] for x_row2 in data]
y_data = [y_row[2] for y_row in data]
print("x1: {}".format(x1))
print("x2: {}".format(x2))
print("y_data: {}".format(y_data))

# 기울기 구하기
a1 = tf.Variable(tf.random_uniform([1], 0, 10, dtype=tf.float64, seed=0))
a2 = tf.Variable(tf.random_uniform([1], 0, 10, dtype=tf.float64, seed=0))
b = tf.Variable(tf.random_uniform([1], 0, 100, dtype=tf.float64, seed=0))

# 다중 방정식
# 예측성적 = x1의 기울기 * x1의값(공부한성적) + x2의 기울기 * x2의값(과외받은수) + y절편
y = a1 * x1 + a2 * x2 + b

# 텐서플로우 RMSE함수 (평균제곱근오차)
rmse = tf.sqrt(tf.reduce_mean(tf.square(y - y_data)))

# 학습률 값
learning_rate = 0.1

# 경사하강법 텐서플로우 함수
gradient_decent = tf.train.GradientDescentOptimizer(learning_rate).minimize(rmse)

# 학습진행
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        sess.run(gradient_decent)
        if step % 10 == 0:
            print("Epoch: %.f, RMSE = %.04f, 기울기a1 = %.4f, 기울기a2 = %.4f, y절편 b = %.4f" %
                  (step, sess.run(rmse), sess.run(a1), sess.run(a2), sess.run(b)))
