# (경사하강법)
import tensorflow as tf

# data[[공부한시간, 실제성적]]
# x_data = 공부한시간
# y_data = 실제성적
data = [[2, 81], [4, 93], [6, 91], [8, 97]]
x_data = [x_row[0] for x_row in data]
y_data = [y_row[1] for y_row in data]
print("x_data : {}".format(x_data))
print("y_data : {}".format(y_data))

# 학습률 값
learning_rate = 0.1

# 기울기 a는 0~10사이 임의값으로
# y절편 b는 0~100사이 임의값으로
# 텐서플로우의 random_uniform()함수로 임의의수를 생성해줌
# random_uniform(뽑아낼 값의수, 최소값, 최대값, 데이터형식(float64), 실행시 같은값이 나올수있게 seed값 설정)
a = tf.Variable(tf.random_uniform([1], 0, 10, dtype=tf.float64, seed=0))
b = tf.Variable(tf.random_uniform([1], 0, 100, dtype=tf.float64, seed=0))

# 일차방정식. y(예측한성적) = a(기울기) * x(공부한시간) + b(y절편)
y = a * x_data + b

# 평균 제곱근 오차.
# 수식1: y(예측성적) - y_data(실제성적)
# 수식2: tf.square(제곱).  y(예측성적) - y_data(실제성적) ^2
# 수식3: tf.reduce_mean(평균값계산) tf.reduce_mean((y(예측성적) - y_data(실제성적) ^2))
# 수식4: tf.sqrt(제곱근) tf.sqrt(tf.reduce_mean((y(예측성적) - y_data(실제성적) ^2)))
rmse = tf.sqrt(tf.reduce_mean(tf.square(y - y_data)))

# GradientDescentOptimizer()함수는 텐서플로우 경사하강법 수식
gradient_decent = tf.train.GradientDescentOptimizer(learning_rate).minimize(rmse)

# 텐서플로우 session함수로 필요한리소스를 할당시키고 실행시킬 준비를 함
# session을 통해 구현될 함수를 '그래프'라고 부름
# session.run('그래프명')으로 함수실행
# tf.global_variables_initializer는 변수초기화 함수. 변수를 초기화시키고
# gradient_decent을 2001번 실행시키고 100번마다 결과를 출력시킴
with tf.Session() as sess:
    # 변수 초기화
    sess.run(tf.global_variables_initializer())
    # 2001번 실행(0 ~ 2000. 총2001번)
    for step in range(2001):
        sess.run(gradient_decent)
        # 100번마다 결과출력
        if step % 100 == 0:
            print(
                "Epoch: %.f, RMSE = %.f, 기울기a = %.4f, y절편 b = %.4f" % (step, sess.run(rmse), sess.run(a), sess.run(b)))
