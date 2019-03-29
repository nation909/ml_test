from IPython import display
from keras.preprocessing import sequence
from konlpy.tag import Twitter
from numpy.ma import argmax
from tensorflow.python.keras.models import load_model
import numpy as np
import pandas as pd

x = [[11, 4, 1309, 43, 22, 67, 36, 82, 24, 1, 16, 303, 391, 7, 276, 95, 35, 16, 12, 55, 53, 44, 36, 574, 65, 1, 55, 14,
      89, 247, 5, 61, 1, 104, 142, 41, 447, 5, 961, 14, 483, 64, 14, 16, 203, 36, 3, 1, 61, 80, 5, 2, 19, 8, 7, 6997,
      421, 1, 30, 1, 42, 72, 81, 2, 17, 151, 2, 144, 3, 61, 61, 2, 101, 1, 61, 487, 277, 110, 27, 328, 102, 41, 5, 483,
      276, 64, 1, 14, 76, 55, 36, 3, 16, 303, 1, 192, 32, 53, 6, 99, 3, 725, 5221, 15510, 868, 6, 45, 101, 3, 6, 4150,
      31, 6, 294, 163, 344, 83, 147, 34, 51, 54, 143, 106, 6, 77, 199, 132, 131, 2001, 1, 25, 57, 1351, 446, 1188, 316,
      1401, 131, 1, 357]]

# 상담콜 전처리 데이터 csv 로드
data = pd.read_csv('../dataset/calldata_csv/20190329/call_preprocessing.csv', encoding='euc-kr', delimiter=',')
input_callText = data['STT_CONT'][0]
result_true = data['CALL_L_CLASS_CD'][0]

twitter = Twitter()
stopwords = ['네네', '디큐']  # 금지어
nounLength = 2  # okt 명사추출 글자수 이상
contNouns = []
tempList = []

print("원본 상담콜: ", input_callText)

tempList = twitter.nouns(input_callText)
contNouns.append([noun for noun in tempList if noun not in stopwords])
print("상담콜 명사 추출: ", contNouns)

#  converters={"nounsAllIndex": lambda x: x.strip("[]").replace("'", "").split(", ")}
call_result = pd.read_csv('../dataset/calldata_csv/20190329/call_result.csv', encoding='euc-kr', delimiter=',',
                          converters={"nounsAllIndex": lambda x: dict(x).strip("[]").replace("'", "").split(", ")})
print("call_result.head(5): ", call_result.head(5))
print("call_result.info(): ", call_result.info())
nounsIndex = call_result['nounsAllIndex']
# print("nounsIndex: ", type(nounsIndex), nounsIndex)
for i in nounsIndex:
    print("i: ", type(i), dict(i))

# 모델 로드
# model = load_model('./model/call_lstm_model.hdf5')

# input 데이터 상담유형 예측
# input_callText = sequence.pad_sequences(contNouns[0])
# result_predict = model.predict(input_callText)
# print("input_callText: ", x)
# print("x_test: ", input_callText)
# print("result_predict: ", result_predict)
#
# result_predict = argmax(result_predict)
# print("예측유형: %d" % result_predict)
# print(result_predict[0][argmax])
