# -*- coding: utf-8 -*-

from __future__ import division, unicode_literals

import re
import itertools
import networkx
import collections

from konlpy.tag import Kkma

_kkma = Kkma()
_stopwords = []


# split으로 문장별로 쪼개기
def xplit(value):
    return re.split('(?:(?<=[^0-9])\.|\n)', value)


def get_sentences(text):
    # text를 양쪽공백제거후 split으로 배열에 문장별로 담음
    candidates = xplit(text.strip())
    sentences = []
    index = 0
    # 문장별로 루프를 진행하면서 문장글자수가 0이거나 마지막글자가 마침표(.)이거나 공백이면 마침표(.)와 공백제거
    for candidate in candidates:
        while len(candidate) and (candidate[-1] == '.' or candidate[-1] == ' '):
            candidate = candidate.strip(' ').strip('.')
        if len(candidate):
            # 문장이 있으면 sentences 배열에 append시키고 index +1추가
            sentences.append(Sentence(candidate + '.', index))
            index += 1
    return sentences


def build_graph(sentences):
    graph = networkx.Graph()
    graph.add_nodes_from(sentences)
    pairs = list(itertools.combinations(sentences, 2))
    for pair in pairs:
        weight = co_occurrence(pair[0], pair[1])
        if weight:
            graph.add_edge(pair[0], pair[1], weight=weight)
    return graph


# p: 동시출현명사들 합, q:전체출현명사들 합
def co_occurrence(sentence1, sentence2):
    p = sum((sentence1.bow & sentence2.bow).values())
    q = sum((sentence1.bow | sentence2.bow).values())
    return p / q if q else 0


class Sentence:

    def __init__(self, text, index=0):
        self.index = index
        self.text = text
        # _stopwords(금지어)에 있는 명사제외하고 꼬꼬마에서 명사추출
        self.nouns = [noun for noun in _kkma.nouns(self.text) if noun not in _stopwords]
        # 추출한 명사들 key:value형식으로 명사:명사가나온수로 count
        self.bow = collections.Counter(self.nouns)

    def __unicode__(self):
        return self.text

    def __str__(self):
        return str(self.index)

    def __repr__(self):
        try:
            return self.text.encode('utf-8')
        except:
            return self.text

    def __eq__(self, another):
        return hasattr(another, 'index') and self.index == another.index

    def __hash__(self):
        return self.index


class TextRank:
    # text: 원문 텍스트
    def __init__(self, text):
        self.sentences = get_sentences(text)
        self.graph = build_graph(self.sentences)
        self.pagerank = networkx.pagerank(self.graph, weight='weight')
        self.reordered = sorted(self.pagerank, key=self.pagerank.get, reverse=True)
        self.nouns = []

        for sentence in self.sentences:
            self.nouns += sentence.nouns
        self.bow = collections.Counter(self.nouns)

    def summarize(self, count=3):
        if not hasattr(self, 'reordered'):
            return ""
        candidates = self.reordered[:count]
        candidates = sorted(candidates, key=lambda sentence: sentence.index)
        return '\n'.join([candidate.text for candidate in candidates])

    def summaryKeyword(self, maxKeyword=3):
        bows = dict([kv for kv in dict(self.bow).items()])
        bows = sorted(bows.items(), key=lambda kv: kv[1], reverse=True)[:maxKeyword]
        return [k[0] for k in bows]


if __name__ == '__main__':
    context = "여야 3당 합의안 발표는 불발…한국당 소집요구하기로 한발 물러서 '손혜원 청문회' 입장차는 여전…여야 갈등 소지 잠복 (서울=연합뉴스) 한지훈 김여솔 이동환 기자 = 여야의 극한 대치로 올해 들어 폐업 상태였던 국회가 4일 정상화 계기를 맞았다. 여야 3당 교섭단체 원내대표는 이날 회동에서 주요 현안에 대한 이견을 좁히지 못했으나, 자유한국당이 돌연 3월 임시국회 소집요구서를 내기로 하면서 파행 국면이 봉합됐다. 이에 따라 3월 국회가 곧 열릴 것으로 보이나, 목포 부동산 투기 의혹을 받는 무소속 손혜원 의원에 대한 야당의 청문회 개최 요구 등 쟁점이 남아 세부 의사일정 합의를 포함한 원활한 국회운영 여부가 주목된다. 더불어민주당 홍영표·자유한국당 나경원·바른미래당 김관영 원내대표는 국회에서 비공개로 만나 3월 임시국회 개회 방안을 논의했으나 별다른 합의안 발표 없이 30여분만에 해산했다. 원내대표들은 '손혜원 청문회' 등 핵심 쟁점을 두고 서로 물러서지 않으면서도 3월 국회를 개회해야 할 때라는 데에는 공감한 것으로 전해졌다. 이에 나경원 원내대표는 회동 직후 기자간담회를 열어 '저희 스스로 결단을 내려 국회를 열기로 했다. 오늘 안에 국회 소집요구서를 내겠다'며 '책임 있는 야당으로서 더 이상 여당에 기대할 게 없다는 생각으로 결단을 내리기로 했다'고 밝혔다. 여당이 손혜원 청문회 등 일련의 조건을 수용하지 않으면 국회 보이콧을 풀 수 없다는 기존의 강경한 입장에서 한 발 물러선 셈이다. 나 원내대표는 '사실 민생을 챙겨야 하는 1차 책임은 정부·여당에 있다'며 '그러나 지금 여당은 그 책임마저 방기하고 자신들의 잘못을 가리는 데 급급하고 자신들의 비리를 감추는 데만 급급하다'고 민주당을 비판했다. 홍 원내대표도 기자들과 만나 '원내대표 회동에서 주요 현안과 일정에 대해 합의를 도출하지 못했다'며 '그러나 방금 나 원내대표가 국회를 소집하겠다는 의사를 밝혀왔다'고 설명했다. 그러면서 '나 원내대표의 결단을 높게 평가하고, 국회가 정상화돼서 늦었지만 다행'이라며 '3월 국회를 통해 그동안 미뤄왔던 시급한 민생입법, 개혁입법을 최대한 빨리 처리해 국회가 일하는 국회로 다시 정상화될 수 있도록 최선을 다하겠다'고 강조했다. 민주당이 한국당의 조건 없는 복귀를 요구하면서 한국당을 뺀 여야 4당의 국회 소집까지 검토한 만큼 한국당의 소집요구서 제출은 사실상 여야 모두가 참여하는 국회 정상화를 의미할 수 있다. 하지만 그동안 교착 정국에서 핵심 쟁점으로 거론됐던 손혜원 국정조사 내지 문화체육관광위원회 차원의 청문회에 대해선 여야 입장차가 여전해 갈등 소지가 가시지 않은 상태다. 여야 간에는 향후 3월 임시국회의 구체적 의사일정을 조율해야 하는 고비도 남아있다. 김관영 원내대표는 회동 후 취재진에게 '한국당이 제가 낸 중재안(손혜원 청문회)을 수용하겠다는 것까지 됐지만, 민주당이 여전히 조건 없이 국회를 열자는 입장'이라고 전했다."
    textrank = TextRank(context)
    summarize = textrank.summarize()
    summaryKeyword = textrank.summaryKeyword()
    print("원문: {}".format(context))
    print("요약: {}".format(summarize))
    print("요악 키워드: {}".format(summaryKeyword))
