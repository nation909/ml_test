from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
import mglearn

X, y = mglearn.datasets.make_wave(n_samples=40)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

reg = KNeighborsRegressor(n_neighbors=3)

reg.fit(X_train, y_train)

print("테스트 세트 예측:{}".format(reg.predict(X_test)))
print("테스트 세트: {:.2f}%".format(reg.score(X_test, y_test) * 100))