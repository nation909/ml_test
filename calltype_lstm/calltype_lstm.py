# -*- coding: utf-8 -*-
from itertools import count

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
from konlpy.tag._okt import Twitter

# 샘플데이터 전처리

# csv 데이터 로드
data = pd.read_csv('../dataset/calltype_data.csv', encoding='euc-kr')
# print(data.head(5))
# print(data['STT_CONT'])
# print(data['STT_CONT'][0])

# csv 전처리(불필요 컬럼삭제, 컬럼수정 등)
data['CALL_LM_CLASS_NAME'] = data['CALL_L_CLASS_NAME'] + "^" + data['CALL_M_CLASS_NAME']
print(data.head(5))
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

# csv 데이터 okt로 명사 추출
twitter = Twitter()


# 상담콜 텍스트 명사 저장 변수


# for cont in data['STT_CONT'][0]:
#     contNounsExtract(cont)

stt_cont_nouns = []
class DataPreprocessing:

    def contNounsExtract(self, cont):
        print("cont: ", cont)
        for text in cont[:5]:
            stt_cont_nouns.append(twitter.nouns(text))


preprcs = DataPreprocessing()
preprcs.contNounsExtract(data['STT_CONT'])
print(stt_cont_nouns)

# print("stt_cont_nouns: ", stt_cont_nouns)
# print("data['STT_CONT'][0]: ", data['STT_CONT'][0])

# X_train = data['STT_CONT']
# Y_train = data['CALL_LM_CLASS_NAME']
# print("X_train: ", X_train)
# print("Y_train: ", Y_train)


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
