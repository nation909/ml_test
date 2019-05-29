import pandas as pd
from keras.preprocessing import sequence
from konlpy.tag import Twitter
from numpy.ma import argmax
from tensorflow.python.keras.models import load_model

# LSTM 모델 사용하기 py
# 상담콜 전처리 데이터 csv 로드
data = pd.read_csv('../dataset/calldata_csv/20190329/call_preprocessing.csv', encoding='euc-kr', delimiter=',')
input_callText = data['STT_CONT'][131]
result_true = data['CALL_L_CLASS_CD'][131]
print("원본 상담콜: ", input_callText)

twitter = Twitter()
stopwords = ['네네', '디큐']  # 금지어
nounLength = 2  # okt 명사추출 글자수 이상
callWordNum = 200

# 상담콜 명사 index 데이터 csv 로드
call_result = pd.read_csv('../dataset/calldata_csv/20190329/call_result.csv', encoding='euc-kr', delimiter=',')
nounsIndex = call_result['nounsAllIndex']

# 명사 사전 index 형변환
nounsIndexDict = {}
for i in nounsIndex:
    temp = i.strip("{}").replace("'", "").split(": ")
    nounsIndexDict[temp[0]] = int(temp[1])
print("명사 사전 index: ", nounsIndexDict)

# 상담콜 명사 추출
contNouns = []
tempList = []
keywords = twitter.nouns(input_callText)
for i in keywords:
    if len(i) >= nounLength:
        tempList.append(i)
contNouns.append([noun for noun in tempList if noun not in stopwords])
print("상담콜 명사 추출: ", contNouns)

# 상담콜 명사 index 추출
input_callIndex = []
for i in contNouns[0]:
    for keyword in nounsIndexDict.keys():
        if i == keyword:
            input_callIndex.append(nounsIndexDict.get(i))
print("상담콜 명사 index 추출: ", type(input_callIndex), input_callIndex)

# 모델 로드
model = load_model('./model/call_lstm_model.hdf5')

# input 데이터 상담유형 예측
x_test = sequence.pad_sequences([input_callIndex], maxlen=callWordNum)
# print("input_callIndex: ", x_test)
result_predict = model.predict(x_test)
print("예측 유형: ", result_predict)

calltype = argmax(result_predict)
print("정답유형: %d, 예측유형: %d, 예측율: %.2f%%" % (result_true, calltype, (result_predict[0][calltype] * 100)))
