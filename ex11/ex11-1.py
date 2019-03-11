# 과적합 피하기
# ex11.py의 학습셋으로 전부 구성이 아닌
# 학습셋 70%, 테스트셋 30%로 구분 후 테스트 진행

from keras.models import Sequential
from keras.layers.core import Dense
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy
import tensorflow as tf
from keras.models import load_model

# seed값 설정
seed = 0
numpy.random.seed(seed)
tf.set_random_seed(seed)
df = pd.read_csv('../dataset/sonar.csv', header=None)

dataset = df.values
print("dataset: {}".format(dataset))
X = dataset[:, 0:60]
Y_obj = dataset[:, 60]
print("X: {}".format(X))
print("Y_obj: {}".format(Y_obj))

e = LabelEncoder()
e.fit(Y_obj)
Y = e.transform(Y_obj)
print("Y: {}".format(Y))

# 학습셋과 테스트셋 구분(테스트셋 30%)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=seed)
print("X_train: {}".format(X_train))  # X 학습셋
print("X_test: {}".format(X_test))  # X 테스트셋
print("Y_train: {}".format(Y_train))  # Y 학습셋
print("Y_test: {}".format(Y_test))  # Y 테스트셋

model = Sequential()
model.add(Dense(24, input_dim=60, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=["accuracy"])
model.fit(X_train, Y_train, epochs=130, batch_size=5)

# 모델 저장
model.save('my_model.h5')

# 모델 삭제 후 저장한 모델 불러오기
del model
model = load_model('my_model.h5')

print("정확도 %.4f" % (model.evaluate(X_test, Y_test)[1]))
