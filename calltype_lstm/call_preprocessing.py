# -*- coding: utf-8 -*-
import collections
from datetime import time
from time import time
import pandas as pd
from konlpy.tag import Twitter, Kkma
from pandas import Series

# 데이터 전처리 py
# csv 데이터 okt로 명사 추출
twitter = Twitter()
kkma = Kkma()
_stopwords = ['네네', '디큐']  # 금지어
nounLength = 2  # okt 명사추출 글자수 이상

# 상담콜별 명사 추출
def contNounsExtract(cont, nounLength=2):
    # print("def contNounsExtract: ", cont)
    print("===========================1단계: 상담콜별 명사추출 start===========================")
    startTime = time()
    contNouns = []
    index = 1
    for text in cont[:1000]:
    # for text in cont:
        index += 1
        try:
            tempList = []
            keywords = twitter.nouns(text)
            if len(keywords) > 0:
                for i in keywords:
                    if len(i) >= nounLength:
                        tempList.append(i)
            else:
                print("keywords 예외건 상담콜: ", text, index)
            contNouns.append([noun for noun in tempList if noun not in _stopwords])

        except Exception as e:
            print("error text, index: ", text, index)
            print(e)

        if index % 1000 == 0:
            print("===========================상담콜 명사추출 %d건 완료===========================" % index)
    print("명사추출 완료 %d건" % len(contNouns))
    print("확인: ", contNouns[:10])
    endTime = time()
    print("상담콜별 명사 추출 Time: %.3f" % (endTime - startTime))
    print("===========================1단계: 상담콜별 명사추출 end===========================")
    return contNouns

# 상담콜별 명사 빈도수 추출
def contNounsCount(contNouns):
    # print("def contNounsCount: ", contNouns)
    print("===========================2단계: 상담콜별 명사 빈도수 추출 start===========================")
    startTime = time()
    contNounsCount = []

    for nouns in contNouns:
        contNounsCount.append(dict(collections.Counter(nouns)))

    endTime = time()
    print("확인: ", contNounsCount[:10])
    print("상담콜별 명사 빈도수 추출 Time: %.3f" % (endTime - startTime))
    print("===========================2단계: 상담콜별 명사 빈도수 추출 end===========================")
    return contNounsCount

# 전체 상담콜 명사 카운트
def contNounsAllCount(contNouns):
    # print("def contNounsAllCount:", contNouns)
    print("===========================3단계: 전체 상담콜 명사 카운트 start===========================")
    startTime = time()
    nounsAllCountDict = {}
    for i in contNouns:
        for k in i:
            if k in nounsAllCountDict:
                nounsAllCountDict[k] = nounsAllCountDict[k] + i[k]
            else:
                nounsAllCountDict[k] = i[k]

    nounsAllCountDict = dict(sorted(nounsAllCountDict.items(), key=lambda kv: kv[1], reverse=True))
    endTime = time()
    print("확인: ", nounsAllCountDict)
    print("상담콜전체 명사 카운트 Time: %.3f" % (endTime - startTime))
    print("===========================3단계: 전체 상담콜 명사 카운트 end===========================")
    return nounsAllCountDict

# 전체문장 명사 index 추출
def nounsAllIndex(nounsAllCount):
    # print("def nounsAllIndex: ", nounsAllCount)
    print("===========================4단계: 전체상담콜 명사 index 추출 start===========================")
    startTime = time()
    nounsAllIndex = {}
    index = 1
    for i in nounsAllCount:
        nounsAllIndex[i] = index
        index += 1

    endTime = time()
    print("확인: ", nounsAllIndex)
    print("전체문장 명사 index 추출 Time: %.3f" % (endTime - startTime))
    print("===========================4단계: 전체상담콜 명사 index 추출 end===========================")
    return nounsAllIndex

# 상담콜별 명사 index 추출
def nounsIndex(nouns, nounsAllIndex):
    # print("def nounsIndex: ", nouns, nounsAllIndex)
    nounIndexList = []

    for noun in nouns:
        if noun in nounsAllIndex.keys():
            nounIndexList.append(nounsAllIndex.get(noun))
    return nounIndexList


class DataPreprocessing:

    def __init__(self, cont):
        print("def __init__ start")
        self.sentence = cont  # 전체문장
        self.nouns = contNounsExtract(cont, nounLength)  # 상담콜별 문장별 명사 추출
        # print("self.nouns: ", self.nouns)

        self.nounsCount = contNounsCount(self.nouns)  # 상담콜별 문장별 명사 빈도수 추출
        # print("self.nounsCount: ", self.nounsCount)

        self.nounsAllCount = contNounsAllCount(self.nounsCount)  # 전체 상담콜 명사 빈도수 카운트 내림차순 정렬 EX. {명사:count수}
        # print("self.nounsAllCount: ", self.nounsAllCount)

        self.nounsAllIndex = nounsAllIndex(self.nounsAllCount)  # 상담콜별 명사 index로 변환 EX. {명사:index번호}
        # print("self.nounsAllIndex: ", self.nounsAllIndex)

        print("===========================5단계: 상담콜별 명사 index 추출 start===========================")
        startTime = time()
        self.nounsIndexList = []  # 문장별 명사 index
        for nouns in self.nouns:
            self.nounsIndexList.append(nounsIndex(nouns, self.nounsAllIndex))
        print("확인: ", self.nounsIndexList[:10])
        # print("self.nounsIndexList: ", self.nounsIndexList)
        endTime = time()
        print("문장별 명사 index 추출 Time: %.3f" % (endTime - startTime))
        print("===========================5단계: 상담콜별 명사 index 추출 end===========================")
        print("def __init__ end")


# csv 데이터 로드
data = pd.read_csv('../dataset/calldata_csv/original_calldata/calldata_original.csv', encoding='euc-kr')
# csv 전처리(불필요 컬럼삭제, 컬럼수정 등)
data['CALL_LM_CLASS_NAME'] = data['CALL_L_CLASS_NAME'] + "^" + data['CALL_M_CLASS_NAME']
del data['RECORDKEY']
# del data['CALL_L_CLASS_CD']
del data['CALL_M_CLASS_CD']
del data['CALL_START_TIME']
del data['CALL_END_TIME']
del data['CALL_L_CLASS_NAME']
del data['CALL_M_CLASS_NAME']
print(data.head(5))
# print("데이터건수: %d" % len(data))

X_train = data['STT_CONT']
Y_train = data['CALL_LM_CLASS_NAME']

preprcs = DataPreprocessing(X_train)
nounsAllCount = preprcs.nounsAllCount
nounsAllIndex = preprcs.nounsAllIndex
nounsIndexList = preprcs.nounsIndexList
# print("전체문장 명사 count수", nounsAllCount)
# print("전체 문장 인덱스 리스트", nounsAllIndex)
# print("문장별 인덱스 리스트: ", nounsIndexList)

data['STT_CONT_INDEX'] = Series(nounsIndexList)
data.to_csv('../dataset/call_preprocessing.csv', index=False, encoding="euc-kr", mode="w", sep=",")

nounsAllCountList = []
for k, v in nounsAllCount.items():
    tempDict = {}
    tempDict[k] = v
    nounsAllCountList.append(tempDict)

nounsAllIndexList = []
for k, v in nounsAllIndex.items():
    tempDict = {}
    tempDict[k] = v
    nounsAllIndexList.append(tempDict)

result = pd.DataFrame()
result['nounsAllCount'] = Series(nounsAllCountList)
result['nounsAllIndex'] = Series(nounsAllIndexList)
result.to_csv('../dataset/call_result.csv', index=False, encoding="euc-kr", mode="w", sep=",")
