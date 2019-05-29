from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
import mglearn

X, y = mglearn.datasets.make_wave(n_samples=60)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
print("X_train: ", X_train, len(X_train))
print("y_train: ", y_train, len(y_train))

ridge = Ridge().fit(X_train, y_train)
print("훈련 세트 점수: {:.2f}".format(ridge.score(X_train, y_train)))
