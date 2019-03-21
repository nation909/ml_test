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