# 로이터뉴스 RNN 활용
# 11,228개의 뉴스기사가 46개의 카테고리로 분류되어 있는 대용량 텍스트 데이터

# 로이터뉴스 데이터 import
from keras.datasets import reuters
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding
from keras.preprocessing import sequence
from keras.utils import np_utils

import numpy
import tensorflow as tf
import matplotlib.pyplot as plt

# 시드설정
seed = 0
numpy.random.seed(seed)
tf.set_random_seed(seed)

# 로이터뉴스 데이터 불러오기
# num_words: 빈도가 높은 단어만 불러옴. 1~1000에 해당하는 단어만 불러옴
# test_split: 테스트셋으로 사용할 데이터. 20%
(X_train, Y_train), (X_test, Y_test) = reuters.load_data(num_words=100, test_split=0.2)

category = numpy.max(Y_train) + 1
print("카테고리: {}".format(category))
print("학습용 뉴스기사: {}".format(len(X_train)))
print("테스트용 뉴스기사: {}".format(len(X_test)))
print("1번째 뉴스기사 데이터: {}".format(X_train[0]))
print("X_train: ", X_train)
print("Y_train: ", Y_train)
# 각 기사의 단어수가 제각각 다르므로 sequences()함수로 단어의 숫자를 맞춤
# maxlen=100: 단어수를 100개로 맞춤. 100개보다 크면 100개째 단어만 사용, 100개보다 작으면 나머지는 0으로 채움
x_train = sequence.pad_sequences(X_train, maxlen=100)
x_test = sequence.pad_sequences(X_test, maxlen=100)
# 원-핫 인코딩 처리(분류를 0, 1로 변경)
y_train = np_utils.to_categorical(Y_train)
y_test = np_utils.to_categorical(Y_test)
print("x_train: ", x_train)
print("y_train: ", y_train)

# 모델 설정
model = Sequential()
# Embedding()함수: 데이터 전처리과정을 통해 입력된 값을 받아 다음층이 알수 있는 형태로 변환
# 모델 설정부분의 맨 처음에 있어야 함
# Embedding(불러온 단어의 총개수(1000), 기사당 단어수(100))
model.add(Embedding(1000, 100))
# LSTM()함수는 RNN에서 기억값에 대한 가중치를 제어함
# LSTM(기사당 단어수(100), 기타옵션(tanh활성화 함수를 사용)))
model.add(LSTM(100, activation="tanh"))
model.add(Dense(46, activation="softmax"))

# 모델 컴파일
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# 모델 실행
# 20개의 스텝으로 100개씩 학습. x_test, y_test가 테스트셋
history = model.fit(x_train, y_train, epochs=20, batch_size=100, validation_data=(x_test, y_test))

print("정확도: %.4f" % (model.evaluate(x_test, y_test)[1]))
#
# # 학습셋오차, 테스트셋오차 저장
y_loss = history.history['loss']
y_vloss = history.history['val_loss']

# 오차데이터를 차트로 시각화
x_len = numpy.arange(len(y_loss))
plt.plot(x_len, y_loss, marker=".", c="blue", label="trainset loss")
plt.plot(x_len, y_vloss, marker=".", c="red", label="testset loss")

plt.legend(loc="upper right")
plt.grid()
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()
