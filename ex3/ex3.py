import numpy as np

# a[0]은 기울기, a[1]은 y절편
ab = [3, 76]
data = [[2, 81], [4, 93], [6, 91], [8, 97]]

# x: 공부한 시간, y: 실제설적
x = [i[0] for i in data]
y = [i[1] for i in data]


# y값(예측값. 예측성적) 구하기 y = ax + b
def predict(x):
    return ab[0] * x + ab[1]


# 평균제곱근오차 = 루트 1/n(원소의개수) * ((실제값 - 예측값)^2 합들)
# p: 실제값, a: 예측값, sqrt는 제곱근
def rmse(p, a):
    print("평균제곱근오차 구하기1 (실제값 - 예측값): ", (p - a) ** 2)
    print("평균제곱근오차 구하기2 (실제값 - 예측값)^2: ", ((p - a) ** 2).mean())
    print("평균제곱근오차 구하기3: ", np.sqrt(((p - a) ** 2).mean()))
    return np.sqrt(((p - a) ** 2).mean())


# 평균제곱근오차를 y에 대입하여 최종값을 구하는 함수
# predict_result: 예측성적, y: 실제성적
def rmse_val(predict_result, y):
    print(predict_result)
    print(np.array(y))
    return rmse(np.array(predict_result), np.array(y))


# predict_result는 예측성적 담기
predict_result = []

for i in range(len(x)):
    print("공부한시간: %.f, 실제성적: %.f, 예측성적: %.f" % (x[i], y[i], predict(x[i])))
    predict_result.append(predict(x[i]))

print("평균제곱근오차 최종값: {}".format(rmse_val(predict_result, y)))