# 다층 퍼셉트론 예제(XOR 해결하기)

# W(1) = [-2, 2
#         -2, 2]
# B(1) = [3
#        -1]
# W(2) = [1
#         1]
# B(2) = [-1]

# 다층 퍼셉트론 수식
# n1 = 시그모이드함수(x1 * w11 + x2 * w21 + b1)
# n2 = 시그모이드함수(x1 * w12 + x2 * w22 + b2)
# 2식의 결과값이 출력층으로 보내짐
# y = 시그모이드함수(n1 * w31 + n2 * w32 + b3)

# n1의 값은 x1, x2가 모두 1일때 0을 출력하고, 하나라도 0이면 1을 출력함(NAND 게이트)
# n2의 값은 x1, x2가 하나라도 1이면 1을 출력함(OR 게이트)
# y의 값은 n1, n2의 값이 모두 1일때 1을 출력함(AND 게이트)

import tensorflow as tf
import numpy as np

w11 = np.array([-2, -2])
w12 = np.array([2, 2])
w2 = np.array([1, 1])
b1 = 3
b2 = -1
b3 = -1


# 퍼셉트론 함수
def MLP(x, w, b):
    y = np.sum(w * x) + b
    if y <= 0:
        return 0
    else:
        return 1


# NAND 게이트
# n1의 x1, x2의 값이 모두 1일때 0, 하나라도1이면 1
def NAND(x1, x2):
    return MLP(np.array([x1, x2]), w11, b1)


# OR 게이트
# n2의 x1, x2의 값이 하나라도 1이면 1, 둘다 0이면 0
def OR(x1, x2):
    return MLP(np.array([x1, x2]), w12, b2)


# AND 게이트
# y의 n1, n2의 값이 둘다 1이면 1, 아니면 0
def AND(x1, x2):
    return MLP(np.array([x1, x2]), w2, b3)


# XOR 게이트
def XOR(x1, x2):
    return AND(NAND(x1, x2), OR(x1, x2))


if __name__ == '__main__':
    for x in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        y = XOR(x[0], x[1])
        print("입력 값: " + str(x) + "출력 값: " + str(y))
