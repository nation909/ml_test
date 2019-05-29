import os
import pandas as pd
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Embedding, Dense, LSTM
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.utils import np_utils

# LSTM모델 개발 py
# 데이터 불러오기
data = pd.read_csv('../dataset/calldata_csv/20190329/call_preprocessing.csv', encoding='euc-kr', delimiter=',',
                   converters={"STT_CONT_INDEX": lambda x: x.strip("[]").replace("'", "").split(", ")})
allCallWordDict = pd.read_csv('../dataset/calldata_csv/20190329/call_result.csv', encoding='euc-kr',
                              delimiter=',')

allCallWordNum = len(allCallWordDict['nounsAllCount'])
callWordNum = 200
trainSet = 0.7  # 트레이닝셋 %
testSet = 0.3  # 테스트셋 %
batch_size = 100
epochs = 20
patience = 10

dataset = data.values
X_train = dataset[:, 2]
Y_train = dataset[:, 1:2]
n_of_train = int(round(len(X_train) * trainSet))  # 트레이닝셋 개수. 올림
n_of_test = int(round((len(X_train)) * testSet))  # 트레이닝셋 개수. 올림
print("샘플 개수: %d" % len(X_train))
print("트레이닝셋 개수: %d, 트레이닝셋: %d" % (n_of_train, (trainSet * 100)))
print("테스트셋 개수: %d, 테스트셋: %d" % (n_of_test, (testSet * 100)))
print("전체 사용 단어 개수: %d, 콜당 사용 단어 개수: %d" % ((allCallWordNum + 1), callWordNum))

# 트레이닝셋, 테스트셋 설정
x_train = sequence.pad_sequences(X_train[:n_of_train], maxlen=callWordNum)  # 트레이닝셋 x
y_train = np_utils.to_categorical(Y_train[:n_of_train], 14)  # 트레이닝셋 y
x_test = sequence.pad_sequences(X_train[n_of_train:], maxlen=callWordNum)  # 테스트셋 x
y_test = np_utils.to_categorical(Y_train[n_of_train:], 14)  # 테스트셋 y

# 모델 설정
model = Sequential()
model.add(Embedding(allCallWordNum + 1, callWordNum))
model.add(LSTM(callWordNum, activation="tanh"))
model.add(Dense(14, activation="softmax"))

# 모델 컴파일
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

MODEL_DIR = './model/'
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

modelpath = MODEL_DIR + "{epoch: 02d}-{val_loss:.4f}-{acc:.4f}.hdf5"
checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=1, save_best_only=True)

early_stopping_callback = EarlyStopping(monitor='val_loss', patience=patience)

# 모델 실행
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=batch_size,
                    callbacks=[checkpointer, early_stopping_callback])
print("정확도: %.4f" % (model.evaluate(x_test, y_test)[1]))

# 모델 저장
model.save('./model/call_lstm_model.h5')
