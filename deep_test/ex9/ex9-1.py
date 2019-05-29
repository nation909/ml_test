# seed값 생성
import numpy
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense

seed = 0
numpy.random.seed(seed)
tf.set_random_seed(seed)

dataset = numpy.loadtxt("../dataset/pima-indians-diabetes.csv", delimiter=",")
X = dataset[:, 0:8]
Y = dataset[:, 8]

# 모델 설정
model = Sequential()
# 8개의 입력값(정보)로 12개의 노드에 보냄 렐루함수사용
model.add(Dense(12, input_dim=8, activation="relu"))
model.add(Dense(8, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

# 모델 컴파일
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# 모델 실행
model.fit(X, Y, epochs=200, batch_size=10)

print("정확도: %.4f" % (model.evaluate(X, Y)[1]))
