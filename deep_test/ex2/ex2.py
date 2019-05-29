# 공부한 시간으로 성적 예측하기(단순선형회귀)

import numpy as np

# x는 공부한 시간
# y는 실제성적
x = [2, 4, 6, 8]
y = [81, 93, 91, 97]
print("공부한 시간: {}".format(x))
print("실제성적: {}".format(y))

# .mean함수는 평균을 구하는 함수
# mx, my는 각각의 평균값
mx = np.mean(x)
my = np.mean(y)
print("x의 평균값: {}".format(mx))
print("y의 평균값: {}".format(my))

# 1. 기울기 구하기(최소제곱법)
#
#        (x - x평균)*(y - y평균) 합
# a ===  ---------------------------
#               (x - x평균)^2
# a = 기울기

# 분모 수식
divisor = sum([(i - mx) ** 2 for i in x])
print("분모: {}".format(divisor))
# 분자 수식
print([i - mx for i in x])
print([i - my for i in y])


def top(x, mx, y, my):
    d = 0
    for i in range(len(x)):
        d += (x[i] - mx) * (y[i] - my)
    return d


dividend = top(x, mx, y, my)
print("분자: {}".format(dividend))
print("기울기: {}".format(dividend / divisor))

# 기울기
a = dividend / divisor

# 2. 절편 구하기
# b = y평균 - (x평균 * 기울기a)
# b = y절편
print("y절편: {}".format(my - (mx * a)))
b = my - (mx * a)

# 3. y값 구하기
# y = ax + b
# y의 값은 = 기울기*x + y의절편
# x는 공부한 시간
print("실제성적: {}".format(y))
print("공부한 시간으로 성적예측: {}".format([(a * i) + b for i in x]))
