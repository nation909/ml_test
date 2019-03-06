# -*- coding: utf-8 -*-

import re
import itertools
import networkx
import collections
import time
import json
from konlpy.tag import Kkma
from konlpy.tag import Twitter

# _morp = Kkma()
_morp = Twitter()

_stopwords = []


def _format_time(end_time):
    return '{:02d}:{:02d}:{:02d}'.format(end_time // 3600, (end_time % 3600 // 60), end_time % 60)


def xplit(value):
    return re.split('(?:(?<=[^0-9])\.|\n|(?:\;))', value)


def get_sentences(ss, ns):
    sentences = []
    index = 0
    for s, n in zip(ss, ns):
        while len(s) and (s[-1] == '.' or s[-1] == ' '):
            s = s.strip(' ').strip('.')
        if len(s) and len(n):
            sentences.append(Sentence(s + '.', n, index))
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


def co_occurrence(sentence1, sentence2):
    p = sum((sentence1.bow & sentence2.bow).values())
    q = sum((sentence1.bow | sentence2.bow).values())
    return p / q if q else 0


class Sentence:
    def __init__(self, s, n, index=0):
        self.index = index
        self.text = s
        self.nouns = [noun for noun in n if noun not in _stopwords]
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

    def rtn(self):
        return self.weight


class TextRank:
    def __init__(self, ss, ns, sw):
        print('ss : ', ss)
        print('ns : ', ns)
        print('sw : ', sw)
        global _stopwords
        _stopwords = _stopwords + sw
        self.sentences = get_sentences(ss, ns)
        self.graph = build_graph(self.sentences)
        self.pagerank = networkx.pagerank(self.graph, weight='weight')
        self.reordered = sorted(self.pagerank, key=self.pagerank.get, reverse=True)
        self.nouns = []
        for sentence in self.sentences:
            self.nouns += sentence.nouns
        self.bow = collections.Counter(self.nouns)

    def summarize(self, count=2):
        if not hasattr(self, 'reordered'):
            return []
        candidates = self.reordered[:count]
        candidates = sorted(candidates, key=lambda sentence: sentence.index)
        return [candidate.text for candidate in candidates]

    def bows(self):
        return dict(self.bow)

    def bestBows(self, min=1, count=3):
        bows = dict([kv for kv in dict(self.bow).items() if kv[1] > min])
        bows = sorted(bows.items(), key=lambda kv: kv[1], reverse=True)[:count]
        return [b[0] for b in bows]


if __name__ == '__main__':
    context = "자유한국당, 바른미래당, 민주평화당, 정의당 등 야 4당이 20일 공공기관 채용비리에 대한 국정조사를 공식 요구했다. 특히 조명래 환경부 장관 임명에 대한 문재인 대통령의 사과, 조국 민정수석 경질, 채용비리 국정조사를 요구하며 국회 일정을 거부 중인 한국당과 바른미래당은 국정조사만 수용되면 국회를 정상화하겠다고 했다. 여당인 더불어민주당은 당내 여론을 수렴해 야당이 협상안으로 내놓은 ‘국정조사 실시-국회 정상화’ 방안의 수용 여부를 결정키로 했다. 이에 따라 민주당의 결정이 예산국회 정상화의 분수령이 될 것으로 보인다. 민주당 홍영표, 한국당 김성태, 바른미래당 김관영, 평화당 장병완, 정의당 윤소하 원내대표 등 여야 5당 원내대표는 이날 국회에서 문희상 국회의장 주재로 회동을 갖고 국회 정상화 방안을 논의했다. 야 4당은 서울교통공사의 고용세습 의혹과 강원랜드 채용비리 의혹을 함께 조사하는 국정조사 수용을 여당에 요구했다. 한국당, 바른미래당은 여당이 국정조사 요구를 수용하면 국회를 정상화하겠다는 뜻을 전했다. 문 의장도 야 4당의 요구가 있는 만큼 조속한 국회 정상화를 위해 여야가 합의해야 한다고 종용한 것으로 알려졌다. 예산안 심사를 앞두고 멈춰 선 국회를 정상화하기 위해 여당이 대승적으로 국정조사를 수용할 필요가 있다는 뜻을 에둘러 전달한 것이라는 해석이 나온다."
    # context = "여야 3당 합의안 발표는 불발…한국당 소집요구하기로 한발 물러서 '손혜원 청문회' 입장차는 여전…여야 갈등 소지 잠복 (서울=연합뉴스) 한지훈 김여솔 이동환 기자 = 여야의 극한 대치로 올해 들어 폐업 상태였던 국회가 4일 정상화 계기를 맞았다. 여야 3당 교섭단체 원내대표는 이날 회동에서 주요 현안에 대한 이견을 좁히지 못했으나, 자유한국당이 돌연 3월 임시국회 소집요구서를 내기로 하면서 파행 국면이 봉합됐다. 이에 따라 3월 국회가 곧 열릴 것으로 보이나, 목포 부동산 투기 의혹을 받는 무소속 손혜원 의원에 대한 야당의 청문회 개최 요구 등 쟁점이 남아 세부 의사일정 합의를 포함한 원활한 국회운영 여부가 주목된다. 더불어민주당 홍영표·자유한국당 나경원·바른미래당 김관영 원내대표는 국회에서 비공개로 만나 3월 임시국회 개회 방안을 논의했으나 별다른 합의안 발표 없이 30여분만에 해산했다. 원내대표들은 '손혜원 청문회' 등 핵심 쟁점을 두고 서로 물러서지 않으면서도 3월 국회를 개회해야 할 때라는 데에는 공감한 것으로 전해졌다. 이에 나경원 원내대표는 회동 직후 기자간담회를 열어 '저희 스스로 결단을 내려 국회를 열기로 했다. 오늘 안에 국회 소집요구서를 내겠다'며 '책임 있는 야당으로서 더 이상 여당에 기대할 게 없다는 생각으로 결단을 내리기로 했다'고 밝혔다. 여당이 손혜원 청문회 등 일련의 조건을 수용하지 않으면 국회 보이콧을 풀 수 없다는 기존의 강경한 입장에서 한 발 물러선 셈이다. 나 원내대표는 '사실 민생을 챙겨야 하는 1차 책임은 정부·여당에 있다'며 '그러나 지금 여당은 그 책임마저 방기하고 자신들의 잘못을 가리는 데 급급하고 자신들의 비리를 감추는 데만 급급하다'고 민주당을 비판했다. 홍 원내대표도 기자들과 만나 '원내대표 회동에서 주요 현안과 일정에 대해 합의를 도출하지 못했다'며 '그러나 방금 나 원내대표가 국회를 소집하겠다는 의사를 밝혀왔다'고 설명했다. 그러면서 '나 원내대표의 결단을 높게 평가하고, 국회가 정상화돼서 늦었지만 다행'이라며 '3월 국회를 통해 그동안 미뤄왔던 시급한 민생입법, 개혁입법을 최대한 빨리 처리해 국회가 일하는 국회로 다시 정상화될 수 있도록 최선을 다하겠다'고 강조했다. 민주당이 한국당의 조건 없는 복귀를 요구하면서 한국당을 뺀 여야 4당의 국회 소집까지 검토한 만큼 한국당의 소집요구서 제출은 사실상 여야 모두가 참여하는 국회 정상화를 의미할 수 있다. 하지만 그동안 교착 정국에서 핵심 쟁점으로 거론됐던 손혜원 국정조사 내지 문화체육관광위원회 차원의 청문회에 대해선 여야 입장차가 여전해 갈등 소지가 가시지 않은 상태다. 여야 간에는 향후 3월 임시국회의 구체적 의사일정을 조율해야 하는 고비도 남아있다. 김관영 원내대표는 회동 후 취재진에게 '한국당이 제가 낸 중재안(손혜원 청문회)을 수용하겠다는 것까지 됐지만, 민주당이 여전히 조건 없이 국회를 열자는 입장'이라고 전했다."
    ss = xplit(context)
    ns = []
    for s in ss:
        ns.append(_morp.nouns(s))
    # ss = ['꿈을 이루는 케이비 국민 은행 최신입니다 ', '예', '뭐 좀 여쭤볼려고 전 화 드렸는데요.', '네', '예 ', '고객님', '아 까지는', '저는엔 에이치 입금을 했는데']
    # ns = [['꿈', '케이', '비', '국민', '은행', '최신'], ['예'], ['뭐', '좀', '전화'], ['네'], ['예'], ['고객'], [], ['저', '에이치', '입금']]
    tr = TextRank(ss, ns, ['에', '안녕하세요', '국정조사'])
    print(tr.summarize())
    # print(dict(tr.bows()))
    print(tr.bestBows())
