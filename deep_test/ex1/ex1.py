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
Data_set = numpy.loadtxt('../dataset/ThoraricSurgery.csv', delimiter=',')
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
# activation: 다음층으로 어떻게 넘길지 결정(relu, sigmoid함수를 사용)
# 30: 30개의 노드, input_dim=17: 17개의 값. 따라서 17개의 입력값으로 30개의 노드로 보낸다는 의미
# 17개의 입력값으로 임의의 가중치를 갖고 30개의 노드로 전달하여 relu함수를 사용
model.add(Dense(30, input_dim=17, activation='relu'))
# 출력층이므로 1개의 노드를 갖고 시그모이드함수를 사용
model.add(Dense(1, activation='sigmoid'))

# compile을 통해 실행
# loss: 한번 신경망이 실행될 때마다 오차 값을 추적하는 함수
# optimizer: 오차를 어떻게 줄여갈지 정하는 함수
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.fit(X, Y, epochs=50, batch_size=10)

# .evaluate함수로 딥러닝 모델의 정확도를 예측하는지 점검
print("정확도: %.4f " % model.evaluate(X, Y)[1])

# 코드내용
# 기존환자들의 데이터중 일부를 랜덤하게 추출해 새환자인 것으로 가정하고 테스트한 결과

# 용어정리
# 데이터는 폐암수술환자의 생존여부 데이터는 470명 환자에게 17개의 정보를 정리한 것
# 속성: 17개의 정보를 뜻함
# 클래스: 생존여부를 뜻함
# 샘플: 각 환자들의 정보를 뜻함(17개 정보)
# ex. 470개의 샘플이 각각 17개의 속성을 가지고 있음

# 프로세스
# 1. 데이터를 변수에 받음
# 2. 퍼셉트론 위에 층을 추가함(렐루(은닉층), 시그모이드(출력층)
# 3. 모델이 효과적으로 구현될 수 있도록 환경설정을 해줌
# - 오차는 mean_squared_error: 평균제곱오차를 사용. 예측값이 0, 1일경우 binary_crossentropy(이항 교차 엔트로피)를 주로 사용함
# - 최적화는 adam을 사용
# - 측정항목(metrics)은 모델수행결과를 나타내는 설정. accuracy: 정확도를 사용
# 4. 컴파일단계에서 설정한 환경으로 데이터를 불러 실행시킴(모델을 실행) model.fit(X, Y, epochs=30, batch_size=10)
# - epoch: 학습프로세스가 모든샘플에 대해서 한번 실행하는 것을 1epoch라고 부름(스텝)
# - bath_size: 샘플을 한번에 몇개씩 처리할지를 결정하는 부분(470개 샘플을 10개씩 끊어서 처리한다는 의미)
