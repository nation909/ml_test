# 와인
# 0: 주석산 농도
# 1: 아세트산 농도
# 2: 구연산 농도
# 3: 진류 당분 농도
# 4: 염화나트륨 농도
# 5: 유리 아황산 농도
# 6: 총 아황산 농도
# 7: 밀도
# 8: pH
# 9: 황산칼륨 농도
# 10: 알코올 도수
# 11: 와인의 맛(0~10등급)
# 12: class(0: 화이트와인, 1: 레드와인)

import numpy
import pandas as pd
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense
import os
from keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt

seed = 0
numpy.random.seed(seed)
tf.set_random_seed(seed)

df_pre = pd.read_csv('../dataset/wine.csv', header=None)
# sample함수로 원본데이터의 몇%를 사용할지 결정(EX. 1: 100%, 0.5: 50%)
# sample함수는 원본데이터에서 정해진 비율만큼 랜덤으로 뽑아오는 함수(frac=1일경우 원본데이터를 모두 랜덤으로 불러옴)
df = df_pre.sample(frac=0.15)
print(df.head(5))

dataset = df.values
print("dataset: {}".format(dataset))

X = dataset[:, 0:12]
Y = dataset[:, 12]
print("X: {}".format(X))
print("Y: {}".format(Y))

# 모델 설정
model = Sequential()
model.add(Dense(30, input_dim=12, activation="relu"))
model.add(Dense(12, activation="relu"))
model.add(Dense(8, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

# 모델 컴파일
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
print("정확도: %.4f" % (model.evaluate(X, Y)[1]))

MODEL_DIR = './model/'
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

modelpath = MODEL_DIR + '{epoch:02d}-{val_loss:.4f}.hdf5'

# history변수에 모델이 학습되는 과정을 저장하고
# 전체샘플중 15%만 불러오고
# 불러온샘플중 33%는 분리하여 테스트셋으로 사용
history = model.fit(X, Y, validation_split=0.33, epochs=3500, batch_size=500)

# y_vloss: 테스트오차값 저장
y_vloss = history.history['val_loss']
# y_acc: 정확도 저장
y_acc = history.history['acc']

# x값 지정
x_len = numpy.arange(len(y_acc))
print("x_len: {}".format(x_len))
# 테스트오차값은 빨강, 정확도는 파랑색으로 표시
plt.plot(x_len, y_vloss, "o", c="red", markersize=3)
plt.plot(x_len, y_acc, "o", c="blue", markersize=3)

plt.show()
