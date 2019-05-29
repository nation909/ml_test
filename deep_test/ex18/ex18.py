# LSTM과 CNN의 조합으로 영화리뷰 분류하기

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding, LSTM, Conv1D, MaxPooling1D
from keras.datasets import imdb

import numpy
import tensorflow as tf
import matplotlib.pyplot as plt

# 시드 설정
seed = 0
numpy.random.seed(seed)
tf.set_random_seed(seed)

# 1~5000까지의 빈도의 단어만 불러옴
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=5000)

# 100개의 단어수만 가져옴
x_train = sequence.pad_sequences(x_train, maxlen=100)
x_test = sequence.pad_sequences(x_test, maxlen=100)

# 모델 설정
model = Sequential()
# 다음층이 알 수 있도록 5000개의 불러온 단어와 데이터들의 단어 100개를 형태 변환
model.add(Embedding(5000, 100))
# 50%의 노드를 드롭아웃시킴
model.add((Dropout(0.5)))
# 컨볼루션함수로 64개의 마스크를 사용
model.add((Conv1D(64, 5, padding="valid", activation="relu", strides=1)))
# 맥스풀링함수로 축소시킴 4
model.add(MaxPooling1D(pool_size=4))
# LSTM함수로 가중치를 제어함 55
model.add(LSTM(55))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.summary()

# 모델 컴파일
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# 모델 실행
history = model.fit(x_train, y_train, epochs=5, batch_size=100, validation_data=(x_test, y_test))

print("정확도 %.4f" % (model.evaluate(x_test, y_test)[1]))

y_loss = history.history['loss']
y_vloss = history.history['val_loss']

x_len = numpy.arange(len(y_loss))
plt.plot(x_len, y_loss, marker=".", c="blue", label="trainset_loss")
plt.plot(x_len, y_vloss, marker=".", c="red", label="testset_loss")

plt.legend(loc="upper right")
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
