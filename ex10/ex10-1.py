import numpy
import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from keras import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

seed = 0
numpy.random.seed(seed)
tf.set_random_seed(seed)

df = pd.read_csv('../dataset/iris.csv',
                 names=["sepal_length", "sepal_width", "petal_length", "petal_width", "species"])

sns.pairplot(df, hue="species")
plt.show()

dataset = df.values
print(dataset)
# X값: 4개의 정보를 받음
# Y_obj값: 클래스를 받음(EX. 'Iris-setosa', 'Iris-versicolor')
X = dataset[:, 0:4].astype(float)
Y_obj = dataset[:, 4]
print("X: {}".format(X))
print("Y_obj: {}".format(Y_obj))

# 문자열을 숫자로 변환
# LabelEncoder함수를 통해 Y_obj의 문자를 0, 1, 2숫자로 변경함)
e = LabelEncoder()
e.fit(Y_obj)
Y = e.transform(Y_obj)
print("Y: {}".format(Y))

# 0, 1, 2로 변경한 Y값을 0, 1로만 이루어진 형태로 변경함 EX. [1. 0. 0.], [0. 1. 1.]
# 여러개의 Y값을 0, 1로만 이루어진 형태로 바꿔주는 기법을 원-핫 인코딩이라고 함
Y_encoded = np_utils.to_categorical(Y)
print("Y_encoded: {}".format(Y_encoded))

# 모델 설정
model = Sequential()
model.add(Dense(16, input_dim=4, activation="relu"))
# 3개의 노드수, 소프트맥스: 총합이 1인 형태로 바꿔서 계산해주는 함수
# 합계가 1인형태로 변환하면 큰값이 두드러지게 나타나고 작은값은 더 작게 나타남
# 이값이 교차엔트로피를 지나[1. 0. 0.]으로 변화하게 되는 것
model.add(Dense(3, activation="softmax"))

# 모델 컴파일
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# 모델 실행
model.fit(X, Y_encoded, epochs=50, batch_size=1)

# 결과 출력
print("정확도 %.4f" % (model.evaluate(X, Y_encoded)[1]))