import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

dataset = pd.read_csv('hitters.csv')

coefs = []
errors = []
w = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
X = dataset.iloc[:, 0:-2].values
y = dataset.iloc[:, -2].values

lambdas = np.logspace(-2, 10, 200)

for lambd in lambdas:
    clf = Ridge(alpha=lambd)
    clf.fit(X, y)
    coefs.append(clf.coef_)
    errors.append(mean_squared_error(clf.coef_, w))
    # print('Degree : %d Training Error %.3f Test Error %.3f'%(deg,train_rmse[-1],test_rmse[-1]))

plt.figure(figsize=(20, 6))

ax = plt.gca()
ax.plot(lambdas, coefs)
ax.set_xscale('log')
plt.xlabel('alpha')
plt.ylabel('weights')
plt.title('Ridge coefficients as a function of the regularization')
plt.axis('tight')

plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
