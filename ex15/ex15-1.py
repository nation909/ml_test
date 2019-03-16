# 6만개의 학습셋, 만개의 테스트셋

# 1. 학습셋, 테스트셋, 학습셋정답, 테스트셋정답 저장
# 2. 모델 은닉층 설정 model.add()
# - 속성, 은닉층, 활설화함수 설정
# 3. 모델 실행환경 설정 model.compile()
# 4. 모델 학습저장 or 성과향상 없을경우 자동학습중단 설정
# - 오차함수, 최적화함수, 측정항목
# 5. 모델 실행 model.fit() model.evaluate()

import numpy
import tensorflow as tf
import pandas as pd
import os
import matplotlib.pyplot as plt

from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping

seed = 0
numpy.random.seed(seed)
tf.set_random_seed(seed)

# 1. 학습셋, 테스트셋, 학습셋정답, 테스트셋정답 저장
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# 학습셋, 테스트셋을 1차원배열로 변경하고 0~255까지의 값을 0~1사이값으로 변경(케라스 최적의조건으로 구동하기위해)
x_train = x_train.reshape(x_train.shape[0], 784).astype('float32') / 255
x_test = x_test.reshape(x_test.shape[0], 784).astype('float32') / 255
# 원핫 인코딩 방식으로 변경(EX. 1 -> [1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

print("학습셋 %d개" % x_train.shape[0])
print("테스트셋 %d개" % x_test.shape[0])

# 2. 모델 은닉층 설정 model.add()
model = Sequential()
model.add(Dense(512, input_dim=784, activation="relu"))
model.add(Dense(10, activation="softmax"))

# 3. 모델 실행환경 설정 model.compile()
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# 4. 모델 학습저장 or 성과향상 없을경우 자동학습중단 설정
MODEL_DIR = "./model/"
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

modelpath = MODEL_DIR + "{epoch: 02d}-{val_loss: .4f}.hdf5"
checkpointer = ModelCheckpoint(filepath=modelpath, monitor="val_loss", save_best_only=True, verbose=1)
early_stopping_callback = EarlyStopping(monitor="val_loss", patience=10)

# 5. 모델 실행 model.fit() model.evaluate()
history = model.fit(x_train, y_train, batch_size=200, epochs=30, validation_data=(x_test, y_test), verbose=0,
                    callbacks=[early_stopping_callback, checkpointer])

print("정확도: %.4f" % (model.evaluate(x_test, y_test)[1]))

# 모델학습데이터를 통한 시각화차트
# 학습셋오차
y_loss = history.history["loss"]
# 테스트셋오차
y_vloss = history.history["val_loss"]

x_len = numpy.arange(len(y_loss))
print("x_len: {}".format(x_len))
plt.plot(x_len, y_loss, marker=".", c="blue", label="학습셋 오차")
plt.plot(x_len, y_vloss, marker=".", c="red", label="테스트셋 오차")
plt.legend(loc="upper right")
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
