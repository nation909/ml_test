# 숫자 이미지 예측 CNN
import sys
from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.utils import np_utils

(X_train, Y_class_train), (X_test, Y_class_test) = mnist.load_data()
print("X_train: {}".format(X_train))
print("Y_class_train: {}".format(Y_class_train))
print("X_test: {}".format(X_test))
print("Y_class_test: {}".format(Y_class_test))

print("학습셋 이미지수: %d개" % (X_train.shape[0]))
print("테스트셋 이미지수: %d개" % (X_test.shape[0]))

# 1번째 글자하나를 이미지로 출력
# 가로 28 x 28 = 784개 픽셀로 이루어짐
# 각 픽셀마다 밝기정도에 따라 0~255까지 등급을 매김(흰색: 0)
# 하나의 긴 행렬로 이루어진 집합으로 변환됨
# 784개의 픽셀이 속성이 되어 0~9까지 10개 클래스중 하나를 예측하는 문제가 됨
plt.imshow(X_train[0], cmap="Greys")
plt.show()

# 코드로 확인
for x in X_train[0]:
    for i in x:
        sys.stdout.write("%d\t" % i)
    sys.stdout.write("\n")

# 1차원 배열로 변경
X_train = X_train.reshape(X_train.shape[0], 784)

# 케라스는 0~1사이의 값으로 변환한다음 구동할때 가장 최적의 성능을 보임
# 0~255로 되어있는 수를 0~1사이의 값으로 변경하는 과정이 필요. (값 / 255)
# 데이터의 폭이 클 때 적절한 값으로 바꾸는 과정을 데이터정규화라고 함

# 실수로 변경
X_train = X_train.astype('float64')
X_train = X_train / 255

# 테스트셋 X도 변경
X_test = X_test.reshape(X_test.shape[0], 784).astype('float64') / 255

# 실제 값 확인
print("class: %d " % (Y_class_train[0]))

# 실제값0~9까지의 값을 원-핫 인코딩 방식으로 변경이 필요
# EX. 5 -> [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
Y_train = np_utils.to_categorical(Y_class_train, 10)
Y_test = np_utils.to_categorical(Y_class_test, 10)
print("Y_train: {}".format(Y_train))
