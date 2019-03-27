import math

import numpy as np
import pandas as pd
from ast import literal_eval
import re
from keras.layers import Embedding, Dense, LSTM
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import np_utils

data = pd.read_csv('../dataset/call_preprocessing_test.csv', encoding='euc-kr', delimiter=',',
                   converters={"STT_CONT_INDEX": lambda x: x.strip("[]").replace("'", "").split(", ")})
# print(data.head(5))
# print(data.info())

dataset = data.values
# print("dataset: {}".format(dataset))

X_train = dataset[:, 1]
Y_train = dataset[:, 0:1]
# print("X_train: ", X_train)
# print("Y_train: ", Y_train)

trainSet = 0.8  # 트레이닝셋 %
testSet = 0.2  # 테스트셋 %
n_of_train = round(len(X_train) * trainSet)  # 트레이닝셋 개수. 올림
n_of_test = round((len(X_train)) * testSet)  # 트레이닝셋 개수. 올림
print("트레이닝셋 개수: %d, 트레이닝셋: %d" % (n_of_train, (trainSet * 100)))
print("테스트셋 개수: %d, 테스트셋: %d" % (n_of_test, (testSet * 100)))

# 트레이닝셋, 테스트셋 설정
x_train = sequence.pad_sequences(X_train[:n_of_train], maxlen=100)  # 트레이닝셋 x
y_train = np_utils.to_categorical(Y_train[:n_of_train], 14)  # 트레이닝셋 y
x_test = sequence.pad_sequences(X_train[n_of_train:], maxlen=100)  # 테스트셋 x
y_test = np_utils.to_categorical(Y_train[n_of_train:], 14)  # 테스트셋 y

# 모델 설정
model = Sequential()
# Embedding()함수로 데이터 전처리과정을 통해 입력된 값을 받아 다음층이 알수 있는 형태로 변환
# 모델 설정부분의 맨 처음에 있어야 함
# Embedding(불러온 단어의 총개수(1000), 기사당 단어수(100))
model.add(Embedding(1000, 100))
# LSTM()함수는 RNN에서 기억값에 대한 가중치를 제어함
# LSTM(기사당 단어수(100), 기타옵션(tanh활성화 함수를 사용)))
model.add(LSTM(100, activation="tanh"))
model.add(Dense(14, activation="softmax"))

# 모델 컴파일
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# 모델 실행
# 20개의 스텝으로 100개씩 학습. x_test, y_test가 테스트셋
history = model.fit(x_train, y_train, epochs=10, batch_size=5, validation_data=(x_test, y_test))

print("정확도: %.4f" % (model.evaluate(x_test, y_test)[1]))
