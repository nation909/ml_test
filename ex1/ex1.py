# 폐암수술환자 데이터를 불러와서 환자 생존율 예측하기

from keras.models import Sequential
from keras.layers import Dense

import numpy
import tensorflow as tf

# 실행할때마다 같은 결과를 출력하기 위해 설정하는 부분?
seed = 0
numpy.random.seed(seed)
tf.set_random_seed(seed)

# 17가지의 정보(종양유형, 폐활량, 호흡곤란여부, 고통정도 등)의 csv데이터를 불러옴. 18번째는 환자의 생존여부(1:생존, 0:사망)
# 17가지의 정보는 속성, 18번째의 생존여부는 클래스
Data_set = numpy.loadtxt('dataset/ThoraricSurgery.csv', delimiter=',')
# print(Data_set)

# X: csv파일의 0~16번까지를 담음
# Y: csv파일의 17번정보를 담음
X = Data_set[:, 0:17]
Y = Data_set[:, 17]
print("x: {}".format(X))
print("y: {}".format(Y))

# Sequential함수는 딥러닝 구조를 한층한층 쌓아올릴수 있도록 하는 함수
# .add로 층을 차례로 추가 (2층 분석모델)
model = Sequential()
# Dense함수는 각층의 특성을 옵션으로 설정하는 역할
model.add(Dense(30, input_dim=17, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# compile을 통해 실행
# activation: 다음층으로 어떻게 넘길지 결정(relu, sigmoid함수를 사용)
# loss: 한번 신경망이 실행될 때마다 오차 값을 추적하는 함수
# optimizer: 오차를 어떻게 줄여갈지 정하는 함수
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.fit(X, Y, epochs=30, batch_size=10)

# .evaluate함수로 딥러닝 모델의 정확도를 예측하는지 점검
print("정확도: %.4f " % model.evaluate(X, Y)[1])

# 코드내용
# 기존환자들의 데이터중 일부를 랜덤하게 추출해 새환자인 것으로 가정하고 테스트한 결과
