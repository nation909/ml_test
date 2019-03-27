import numpy as np
import pandas as pd
from keras.layers import SimpleRNN, Embedding, Dense, LSTM
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

max_features = 10000
# feature로 사용할 단어의 수
maxlen = 500
# e-mail에서 볼 최대 단어의 수는 500개

data = pd.read_csv("../dataset/spam.csv", encoding='latin1')
print(data[:5])

del data['Unnamed: 2']
del data['Unnamed: 3']
del data['Unnamed: 4']
data['v1'] = data['v1'].replace(['ham', 'spam'], [0, 1])
print(data[:5])

X_train = data['v2']
Y_train = data['v1']
print(len(X_train))
# print(X_train)
print(len(Y_train))
# print(Y_train)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train) #5572개의 행을 가진 X_train의 각 행에 토큰화를 수행
sequences = tokenizer.texts_to_sequences(X_train) #단어를 숫자값, 인덱스로 변환하여 저장
print(sequences[:5])

word_index = tokenizer.word_index
print(word_index)