# 과적합피하기
# 오차역전파 알고리즘을 사용한 신경망이 광석과 돌을 구분하는데 효과적인지 실험


from keras.models import Sequential
from keras.layers.core import Dense
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import numpy
import tensorflow as tf

seed = 0
numpy.random.seed(seed)
tf.set_random_seed(seed)

df = pd.read_csv("../dataset/sonar.csv", header=None)
print("df head: {}".format(df.head()))

dataset = df.values
X = dataset[:, 0:60]
Y_obj = dataset[:, 60]
print("dataset: {}".format(dataset))
print("X: {}".format(X))
print("Y_obj: {}".format(Y_obj))

e = LabelEncoder()
e.fit(Y_obj)
Y = e.transform(Y_obj)
print("Y : {}".format(Y))

model = Sequential()
model.add(Dense(24, input_dim=60, activation="relu"))
model.add(Dense(10, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

model.compile(loss="mean_squared_error", optimizer="adam", metrics=["accuracy"])
model.fit(X, Y, epochs=200, batch_size=5)

print("정확도: %.4f" % (model.evaluate(X, Y)[1]))
