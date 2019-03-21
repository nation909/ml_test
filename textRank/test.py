# -*- coding: utf-8 -*-

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd

data = pd.read_csv('../dataset/calltype_data.csv', encoding='euc-kr')
print(data.head(5))
print(data['STT_CONT'])
print(data['STT_CONT'][0])


tokenizer = Tokenizer()
# tokenizer.fit_on_texts(text)
# sequence = tokenizer.fit_on_sequences(text)
# print(sequence)
