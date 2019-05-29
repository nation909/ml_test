import os

import numpy
import pandas as pd
import tensorflow as tf
from keras import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense

seed = 0
numpy.random.seed(seed)
tf.set_random_seed(seed)

df_pre = pd.read_csv('../dataset/wine.csv', header=None)
df = df_pre.sample(frac=0.15)
dataset = df.values
X = dataset[:, 0:12]
Y = dataset[:, 12]

model = Sequential()
model.add(Dense(30, input_dim=12, activation="relu"))
model.add(Dense(12, activation="relu"))
model.add(Dense(8, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

MODEL_DIF = './model/'
if not os.path.exists(MODEL_DIF):
    os.mkdir(MODEL_DIF)

modelpath = MODEL_DIF + '{epoch: 02d}-{val_loss: .4f}.hdf5'

checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=1, save_best_only=True)

early_stopping_callback = EarlyStopping(monitor='val_loss', patience=100)

# 샘플에서 15%의 샘플을 불러와서 그 중 20%샘플을 테스트셋으로 사용
model.fit(X, Y, validation_split=0.2, epochs=3500, batch_size=500, verbose=0,
          callbacks=[early_stopping_callback, checkpointer])

print("정확도: %.4f" % (model.evaluate(X, Y)[1]))
