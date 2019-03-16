# 초음파 광물 예측하기 k겹 교차검증

from keras.models import Sequential
from keras.layers.core import Dense
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

import numpy
import pandas as pd
import tensorflow as tf

# seed 값 설정
seed = 0
numpy.random.seed(seed)
tf.set_random_seed(seed)

df = pd.read_csv('../dataset/sonar.csv', header=None)

dataset = df.values
X = dataset[:, 0:60]
Y_obj = dataset[:, 60]
print("X: {}".format(X))

e = LabelEncoder()
e.fit(Y_obj)
Y = e.transform(Y_obj)
print("Y: {}".format(Y))

# 10개의 파일로 분할
n_fold = 10
skf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=seed)

accuracy = []

# 위에 설정한 n_fold 수(10개) 만큼 for문으로 반복되게 함
for train, test in skf.split(X, Y):
    print("train: {}".format(train))
    print("test: {}".format(test))
    print("X[train]: {}".format(X[train]))
    print("Y[train]: {}".format(Y[train]))
    print("X[test]: {}".format(Y[test]))
    print("Y[test]: {}".format(Y[test]))
    model = Sequential()
    model.add(Dense(24, input_dim=60, activation="relu"))
    model.add(Dense(10, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss="mean_squared_error", optimizer="adam", metrics=["accuracy"])
    model.fit(X[train], Y[train], epochs=100, batch_size=5)
    k_accuracy = "%.4f" % (model.evaluate(X[test], Y[test])[1])
    accuracy.append(k_accuracy)

print("fold 정확도: %.4f" % accuracy)
