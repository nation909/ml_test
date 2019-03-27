# -*- coding: utf-8 -*-
import collections
from datetime import time
from time import time
import pandas as pd
from pandas import Series
from konlpy.tag import Twitter

# 샘플데이터 전처리


# csv 데이터 okt로 명사 추출
twitter = Twitter()
_stopwords = ['네네', '인터넷', '디큐', '면세점', '음음']  # 금지어
nounLength = 2  # okt 명사추출 글자수 이상
maxNouns = 1000  # 상담콜전체 명사 카운트 개수


# 상담콜별 명사 추출
def contNounsExtract(cont, nounLength=2):
    # print("def contNounsExtract: ", cont)
    startTime = time()
    contNouns = []
    index = 1
    for text in cont[:10000]:
        # for text in cont[:5]:
        try:
            # contNouns.append(twitter.nouns(text))
            index += 1
            tempList = []
            for i in twitter.nouns(text):
                if len(i) >= nounLength:
                    tempList.append(i)
            contNouns.append([noun for noun in tempList if noun not in _stopwords])

        except Exception as e:
            print("error contNouns: ", contNouns)
            print("error text: ", text)
            print(e)

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
def contNounsAllCount(contNouns, maxNouns=1000):
    # print("def contNounsAllCount:", contNouns)
    startTime = time()
    nounsAllCountDict = {}
    for i in contNouns:
        for k in i:
            if k in nounsAllCountDict:
                nounsAllCountDict[k] = nounsAllCountDict[k] + i[k]
            else:
                nounsAllCountDict[k] = i[k]

    nounsAllCountDict = dict(sorted(nounsAllCountDict.items(), key=lambda kv: kv[1], reverse=True)[:maxNouns])
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
        self.nouns = contNounsExtract(cont, nounLength)  # 문장별 명사 추출
        # print("self.nouns: ", self.nouns)

        self.nounsCount = contNounsCount(self.nouns)  # 문장별 명사 count수
        # print("self.nounsCount: ", self.nounsCount)

        self.nounsAllCount = contNounsAllCount(self.nounsCount, maxNouns)  # 전체문장 명사 count수 내림차순 정렬 EX. {명사:count수}
        # print("self.nounsAllCount: ", self.nounsAllCount)

        self.nounsAllIndex = nounsAllIndex(self.nounsAllCount)  # 전체문장 명사 index EX. {명사:index번호}
        # print("self.nounsAllIndex: ", self.nounsAllIndex)

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
print("len(data): ", len(data))

X_train = data['STT_CONT']
Y_train = data['CALL_L_CLASS_CD']

preprcs = DataPreprocessing(X_train)
nounsAllCount = preprcs.nounsAllCount
nounsAllIndex = preprcs.nounsAllIndex
nounsIndexList = preprcs.nounsIndexList
print("전체문장 명사 count수", nounsAllCount)
print("전체 문장 인덱스 리스트", nounsAllIndex)
print("문장별 인덱스 리스트: ", nounsIndexList)

# data = pd.DataFrame(data={"STT_CONT_INDEX": nounsIndexList, "CALL_L_CLASS_CD": data['CALL_L_CLASS_CD']})
data['STT_CONT_INDEX'] = Series(nounsIndexList)
data.to_csv('../dataset/call_preprocessing.csv', index=False, encoding="euc-kr", mode="w", sep=",")

result = pd.DataFrame()
result['nounsAllCount'] = nounsAllCount.items()
result['nounsAllIndex'] = nounsAllIndex.items()
result.to_csv('../dataset/call_result.csv', index=False, encoding="euc-kr", mode="w", sep=",")

