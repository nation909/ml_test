# -*- coding: utf-8 -*-

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
from pandas import DataFrame

data = pd.read_csv('../dataset/calltype_data.csv', encoding='euc-kr')
# print(data.head(5))
# print(data['STT_CONT'])
# print(data['STT_CONT'][0])

data['CALL_LM_CLASS_NAME'] = data['CALL_L_CLASS_NAME'] + "^" + data['CALL_M_CLASS_NAME']
print("data: ", data)

del data['RECORDKEY']
del data['CALL_L_CLASS_CD']
del data['CALL_M_CLASS_CD']
del data['CALL_START_TIME']
del data['CALL_END_TIME']
del data['CALL_L_CLASS_NAME']
del data['CALL_M_CLASS_NAME']
# STT_CONT
# CALL_LM_CLASS_NAME
print(data.head(5))

X_train = data['STT_CONT']
Y_train = data['CALL_LM_CLASS_NAME']
print("X_train: ", X_train)
print("Y_train: ", Y_train)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
sequence = tokenizer.fit_on_sequences(X_train)
print(sequence[:5])


# SQL
# select t1.RECORDKEY, t1.STT_CONT, t3.CALL_L_CLASS_NAME, t3.CALL_M_CLASS_NAME, t2.CALL_L_CLASS_CD, t2.CALL_M_CLASS_CD, t2.CALL_START_TIME, t2.CALL_END_TIME
# from stt_lotte.stt_rst_full t1
# inner join ta_lotte.tb_call_info t2
# on t1.RECORDKEY = t2.CALL_ID
# inner join ta_lotte.tb_consult_type t3
# on t2.CALL_L_CLASS_CD = t3.CALL_L_CLASS_CD
# 	AND t2.CALL_M_CLASS_CD = t3.CALL_M_CLASS_CD
# ;
#
#
# select COUNT(*)
# from stt_lotte.stt_rst_full t1
# inner join ta_lotte.tb_call_info t2
# on t1.RECORDKEY = t2.CALL_ID
# inner join ta_lotte.tb_consult_type t3
# on t2.CALL_L_CLASS_CD = t3.CALL_L_CLASS_CD
# 	AND t2.CALL_M_CLASS_CD = t3.CALL_M_CLASS_CD
# limit 10
# ;
