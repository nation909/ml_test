# 피마인디언 데이터 분석 예제
# 데이터: pima-indians-diabetes.csv (샘플 768개, 속성 8개 데이터)
# 정보1: 과거임신횟수
# 정보2. 포도당부하검사2시간 후 공복혈당농도
# 정보3. 확장기 혈압
# 정보4. 삼두근 피부주름두꼐
# 정보5. 혈청 인슐린
# 정보6. 체질량지수
# 정보7: 당뇨병 가족력
# 정보8: 나이
# 클래스: 당뇨 유무: 1(당뇨), 2(당뇨아님)
import numpy
import pandas as pd
import tensorflow as tf

# read_csv함수로 csv데이터를 불러옴
from keras import Sequential
from keras.layers import Dense

df = pd.read_csv('../dataset/pima-indians-diabetes.csv',
                 names=["pregnant", "plasma", "pressure", "thickness", "insulin", "BMI", "predigree", "age", "class"])

print(df.head(5))
# print(df.info())

# 샘플수(count), 평균(mean), 표준편차(std), 최소값(min), 백분위수(25%, 50%, 75%), 최대값(max)
print(df.describe())
# print(df[["pregnant", "class"]])

# groupby함수로 pregnant정보를 기준으로 구성
# as_index=False는 pregnant옆에 새로운 index를 만들어줌
# mean함수로 평균을 구하고, sort_values함수를 써서 pregnant컬럼을 오름차순으로 정리
print(df[["pregnant", "class"]].groupby(["pregnant"], as_index=False).mean().sort_values(by="pregnant", ascending=True))

import matplotlib.pyplot as plt
import seaborn as sns

# 그래프 크기를 결정
plt.figure(figsize=(12, 12))

# linewidth: 라인넓이, vmax: 색상밝기조절, cmap: 미리정해진matplotlib색상 설정값을 불러옴
sns.heatmap(df.corr(), linewidth=0.1, vmax=0.5, cmap=plt.cm.gist_heat, linecolor="white", annot=True)

# 그래프 노출
plt.show()

grid = sns.FacetGrid(df, col="class")
grid.map(plt.hist, "plasma", bins=10)
plt.show()
