# -*- coding: utf-8 -*-
import collections
import math
from datetime import time
from time import time

import pandas as pd
from konlpy.tag._okt import Twitter

# 샘플데이터 전처리

# csv 데이터 okt로 명사 추출
from pandas import Series

twitter = Twitter()


# 상담콜별 명사 추출
def contNounsExtract(cont):
    # print("def contNounsExtract: ", cont)
    startTime = time()
    contNouns = []

    # for text in cont[:5]:
    for text in cont:
        contNouns.append(twitter.nouns(text))

    endTime = time()
    print("상담콜별 명사 추출 Time: %.3f" % (endTime - startTime))
    return contNouns


# 상담콜별 명사 빈도수 추출
def contNounsCount(contNouns):
    # print("def contNounsCount: ", contNouns)
    startTime = time()
    contNounsCount = []

    for nouns in contNouns:
        contNounsCount.append(dict(collections.Counter(nouns)))

    endTime = time()
    print("상담콜별 명사 빈도수 추출 Time: %.3f" % (endTime - startTime))
    return contNounsCount


# 상담콜전체 명사 카운트
def contNounsAllCount(contNouns):
    # print("def contNounsAllCount:", contNouns)
    startTime = time()
    nounsAllCountDict = {}
    for i in contNouns:
        for k in i:
            # print(k, i[k])
            if k in nounsAllCountDict:
                nounsAllCountDict[k] = nounsAllCountDict[k] + i[k]
            else:
                nounsAllCountDict[k] = i[k]

    nounsAllCountDict = dict(sorted(nounsAllCountDict.items(), key=lambda kv: kv[1], reverse=True))

    endTime = time()
    print("상담콜전체 명사 카운트 Time: %.3f" % (endTime - startTime))
    return nounsAllCountDict


# 전체문장 명사 index 추출
def nounsAllIndex(nounsAllCount):
    # print("def nounsAllIndex: ", nounsAllCount)
    startTime = time()
    nounsAllIndex = {}
    index = 1
    for i in nounsAllCount:
        nounsAllIndex[i] = index
        index += 1

    endTime = time()
    print("전체문장 명사 index 추출 Time: %.3f" % (endTime - startTime))
    return nounsAllIndex


def nounsIndex(nouns, nounsAllIndex):
    # print("def nounsIndex: ", nouns, nounsAllIndex)
    nounIndexList = []

    for noun in nouns:
        if noun in nounsAllIndex.keys():
            # print(noun, nounsAllIndex.get(noun))
            nounIndexList.append(nounsAllIndex.get(noun))

    return nounIndexList


class DataPreprocessing:

    def __init__(self, cont):
        print("def __init__ start")
        self.sentence = cont  # 전체문장
        self.nouns = contNounsExtract(cont)  # 문장별 명사 추출
        # print("self.nouns: ", self.nouns)

        self.nounsCount = contNounsCount(self.nouns)  # 문장별 명사 count수
        # print("self.nounsCount: ", self.nounsCount)

        self.nounsAllCount = contNounsAllCount(self.nounsCount)  # 전체문장 명사 count수 내림차순 정렬 EX. {명사:count수}
        # print("self.nounsAllCount: ", self.nounsAllCount)

        self.nounsAllIndex = nounsAllIndex(self.nounsAllCount)  # 전체문장 명사 index EX. {명사:index번호}
        print("self.nounsAllIndex: ", self.nounsAllIndex)

        startTime = time()
        self.nounsIndexList = []  # 문장별 명사 index
        for nouns in self.nouns:
            self.nounsIndexList.append(nounsIndex(nouns, self.nounsAllIndex))
        # print("self.nounsIndexList: ", self.nounsIndexList)
        endTime = time()
        print("문장별 명사 index 추출 Time: %.3f" % (endTime - startTime))
        print("def __init__ end")


# csv 데이터 로드
data = pd.read_csv('../dataset/calltype_data.csv', encoding='euc-kr')

# csv 전처리(불필요 컬럼삭제, 컬럼수정 등)
# data['CALL_LM_CLASS_NAME'] = data['CALL_L_CLASS_NAME'] + "^" + data['CALL_M_CLASS_NAME']
del data['RECORDKEY']
# del data['CALL_L_CLASS_CD']
del data['CALL_M_CLASS_CD']
del data['CALL_START_TIME']
del data['CALL_END_TIME']
del data['CALL_L_CLASS_NAME']
del data['CALL_M_CLASS_NAME']
print(data.head(5))

print(len(data))
startRow = 0  # 추출할 첫행 수
endRow = 0  # 추출할 마지막행 수
rowNum = 3000  # 한번에 추출할 row개수
fileNum = 1  # 파일개수
loopCount = math.ceil(len(data) / rowNum)  # 루프개수 EX. 15855 / 5000 = 4

for i in range(loopCount):
    startTime = time()

    startRow = endRow
    endRow = rowNum * (i + 1)  # 5000
    if endRow > len(data):
        endRow = len(data)
    dataSplit = data[startRow:endRow]  # 0:5000
    print("startRow: %d endRow: %d" % (startRow, endRow))

    X_train = dataSplit['STT_CONT']
    Y_train = dataSplit['CALL_L_CLASS_CD']
    preprcs = DataPreprocessing(X_train)
    nounsIndexList = preprcs.nounsIndexList
    print(preprcs.nounsIndexList)

    data['STT_CONT_INDEX'] = Series(nounsIndexList)
    data.to_csv('../dataset/call_preprocessing' + "{02d}" + ".csv", index=False, encoding="euc-kr", mode="w")

    endTime = time()
    print("%d번째 전처리 시간 Time: %.3f" % ((i + 1), endTime - startTime))

# X_train = data['STT_CONT']
# Y_train = data['CALL_L_CLASS_CD']
# preprcs = DataPreprocessing(X_train)
# nounsIndexList = preprcs.nounsIndexList
# print(preprcs.nounsIndexList)
#
# data['STT_CONT_INDEX'] = Series(nounsIndexList)
# data.to_csv('../dataset/call_preprocessing.csv', index=False, encoding="euc-kr", mode="w")

# SQL
# select t1.RECORDKEY, t1.STT_CONT, t3.CALL_L_CLASS_NAME, t3.CALL_M_CLASS_NAME, t2.CALL_L_CLASS_CD, t2.CALL_M_CLASS_CD, t2.CALL_START_TIME, t2.CALL_END_TIME
# from stt_lotte.stt_rst_full t1
# inner join ta_lotte.tb_call_info t2
# on t1.RECORDKEY = t2.CALL_ID
# inner join ta_lotte.tb_consult_type t3
# on t2.CALL_L_CLASS_CD = t3.CALL_L_CLASS_CD
#   AND t2.CALL_M_CLASS_CD = t3.CALL_M_CLASS_CD
# ;
#
#
# select COUNT(*)
# from stt_lotte.stt_rst_full t1
# inner join ta_lotte.tb_call_info t2
# on t1.RECORDKEY = t2.CALL_ID
# inner join ta_lotte.tb_consult_type t3
# on t2.CALL_L_CLASS_CD = t3.CALL_L_CLASS_CD
#   AND t2.CALL_M_CLASS_CD = t3.CALL_M_CLASS_CD
# limit 10
# ;