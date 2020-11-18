import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

dataset = pd.read_csv('weekly.csv')

# 1

sns.pairplot(dataset,hue='Direction')

ax=plt.axes()
ax.scatter(x=dataset.index,y=dataset['Volume'])
plt.show()

# 2

X = dataset.iloc[:, 1:-2].values
y = dataset.iloc[:, -1].values

model = LogisticRegression(random_state=42)

model.fit(X, y)

print("Training with all features, whole data. \nWeights on features :" )

for i in range(6):
    print("%s : %.2f"%(list(dataset)[i+1] ,model.coef_[0][i]))

print("Confusion matrix :" )
print(confusion_matrix(y, model.predict(X) ))
print("Accuracy  : %.2f "%model.score(X, y))

# 3

X = dataset.iloc[:, 2:3].values
model.fit(X[:985], y[:985])
print("Training with only lag2, upto 2008." )
print("Accuracy on test data 2009 - 10 : %.2f "%model.score(X[985:], y[985:]))
