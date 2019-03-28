import os

import pandas as pd
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Embedding, Dense, LSTM
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.utils import np_utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

data = pd.read_csv('../dataset/call_preprocessing.csv', encoding='euc-kr', delimiter=',',
                   converters={"STT_CONT_INDEX": lambda x: x.strip("[]").replace("'", "").split(", ")})
data = data[:100]
print(data.head(5))
print(data.info())

allCallWordNum = 10000
callWordNum = 1000
trainSet = 0.8  # 트레이닝셋 %
testSet = 0.2  # 테스트셋 %
batch_size = 10
epochs = 50
patience = 10

dataset = data.values
# print("dataset: {}".format(dataset))

X_train = dataset[:, 2]
Y_train = dataset[:, 1:2]

n_of_train = int(round(len(X_train) * trainSet))  # 트레이닝셋 개수. 올림
n_of_test = int(round((len(X_train)) * testSet))  # 트레이닝셋 개수. 올림
print("샘플 개수: %d" % len(X_train))
print("트레이닝셋 개수: %d, 트레이닝셋: %d" % (n_of_train, (trainSet * 100)))
print("테스트셋 개수: %d, 테스트셋: %d" % (n_of_test, (testSet * 100)))

# 트레이닝셋, 테스트셋 설정
x_train = sequence.pad_sequences(X_train[:n_of_train], maxlen=callWordNum)  # 트레이닝셋 x
y_train = np_utils.to_categorical(Y_train[:n_of_train], 14)  # 트레이닝셋 y
x_test = sequence.pad_sequences(X_train[n_of_train:], maxlen=callWordNum)  # 테스트셋 x
y_test = np_utils.to_categorical(Y_train[n_of_train:], 14)  # 테스트셋 y

# 모델 설정
model = Sequential()
# Embedding()함수로 데이터 전처리과정을 통해 입력된 값을 받아 다음층이 알수 있는 형태로 변환
# 모델 설정부분의 맨 처음에 있어야 함
# Embedding(불러온 단어의 총개수(1000), 상담콜당 단어수(100))
model.add(Embedding(allCallWordNum + 1, callWordNum))
# LSTM()함수는 RNN에서 기억값에 대한 가중치를 제어함
# LSTM(기사당 단어수(100), 기타옵션(tanh활성화 함수를 사용)))
model.add(LSTM(callWordNum, activation="tanh"))
model.add(Dense(14, activation="softmax"))

# 모델 컴파일
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

MODEL_DIR = './model/'
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

modelpath = MODEL_DIR + "{epoch: 02d}-{val_loss: .4f}.hdf5"
checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=1, save_best_only=True)

early_stopping_callback = EarlyStopping(monitor='val_loss', patience=patience)

# 모델 실행
# 20개의 스텝으로 100개씩 학습. x_test, y_test가 테스트셋
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=batch_size,
                    callbacks=[checkpointer, early_stopping_callback])
print("정확도: %.4f" % (model.evaluate(x_test, y_test)[1]))
