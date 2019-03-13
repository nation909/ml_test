# EX. 전체 요약 프로세스 설명
# 784개의 속성을 갖고있음
# -> 32개의 마스크, 3x3으로 렐루로 컨볼루션 사용
# -> 64개의 마스크, 3x3으로 렐루로 컨볼루션 사용
# -> 풀링창의 크기를 2(절반)로 정하고 렐루로 맥스풀링 시킴
# -> 드롭아웃으로 25%의 노드를 드롭아웃시킴
# -> 플래튼으로 2차원배열을 1차원으로 바꿔줌
# -> 128개의 은닉층으로 활성화함수는 렐루 사용
# -> 10개의 출력이고 활성화함수는 소프트맥스 사용
# -> 아담으로 모델 실행환경 구성하여 실행시킴

from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping

import numpy
import tensorflow as tf
import os
import matplotlib.pyplot as plt

seed = 0
numpy.random.seed(seed)
tf.set_random_seed(seed)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# 학습셋, 테스트셋을 1차원배열로 변경하고 0~255까지의 값을 0~1사이값으로 변경(케라스 최적의조건으로 구동하기위해)
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

model = Sequential()
# -> 32개의 마스크, 3x3으로 렐루로 컨볼루션 사용
model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(28, 28, 1), activation="relu"))
# -> 64개의 마스크, 3x3으로 렐루로 컨볼루션 사용
model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
# -> 풀링창의 크기를 2(절반)로 정하고 렐루로 맥스풀링 시킴
model.add(MaxPooling2D(pool_size=2))
# -> 드롭아웃으로 25%의 노드를 드롭아웃시킴
model.add(Dropout(0.25))
# -> 플래튼으로 2차원배열을 1차원으로 바꿔줌
model.add(Flatten())
# -> 128개의 은닉층으로 활성화함수는 렐루 사용
model.add(Dense(128, activation="relu"))
# -> 드롭아웃으로 50%의 노드를 드롭아웃시킴
model.add(Dropout(0.5))
# -> 10개의 출력이고 활성화함수는 소프트맥스 사용
model.add(Dense(10, activation="softmax"))

# -> 아담으로 모델 실행환경 구성하여 실행시킴
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

MODEL_DIR = './model/'
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

modelpath = MODEL_DIR + "{epoch: 02d}-{val_loss: .4f}.hdf5"
checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=1, save_best_only=True)

early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=30, batch_size=200, verbose=0,
                    callbacks=[checkpointer, early_stopping_callback])

print("정확도: %.4f" % (model.evaluate(x_test, y_test)[1]))

# 학습셋 오차, 테스트셋 오차
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
