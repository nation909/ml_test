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

seed = 0
numpy.random.seed(seed)
tf.set_random_seed(seed)

df_pre = pd.read_csv('../dataset/wine.csv', header=None)
# sample함수로 원본데이터의 몇%를 사용할지 결정(EX. 1: 100%, 0.5: 50%)
# sample함수는 원본데이터에서 정해진 비율만큼 랜덤으로 뽑아오는 함수(frac=1일경우 원본데이터를 모두 랜덤으로 불러옴)
df = df_pre.sample(frac=1)
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

# 모델 실행
# model.fit(X, Y, epochs=200, batch_size=200)

print("정확도: %.4f" % (model.evaluate(X, Y)[1]))

import os

MODEL_DIR = './model/'
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

modelpath = MODEL_DIR + '{epoch:02d}-{val_loss:.4f}.hdf5'

from keras.callbacks import ModelCheckpoint

# 케라스 내부 예약어
# 테스트오차: val_loss, 학습셋 정확도: acc, 테스트셋정확도: val_acc, 학습셋 오차: loss
# save_best_only=True: 앞 모델보다 나아졌을때만 저장
checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=1, save_best_only=True)
model.fit(X, Y, validation_split=0.2, epochs=200, batch_size=200, verbose=0, callbacks=[checkpointer])
