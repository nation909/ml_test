import pandas as pd

df = pd.read_csv('../dataset/iris.csv',
                 names=['sepal_length', 'sepal_width', 'pertal_length', 'pertal_width', 'species'])

print(df.head())

import seaborn as sns
import matplotlib.pyplot as plt

sns.pairplot(df, hue="species")
plt.show()