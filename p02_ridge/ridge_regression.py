import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, RidgeCV, LassoCV
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

# plt.figure(figsize=(20, 6))
#
# ax = plt.gca()
# ax.plot(lambdas, coefs)
# ax.set_xscale('log')
# plt.xlabel('alpha')
# plt.ylabel('weights')
# plt.title('Ridge coefficients as a function of the regularization')
# plt.axis('tight')
#
# plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

lambdas = [0,4,1e10]
rmse = []

for lambd in lambdas:
    clf = Ridge(alpha=lambd)
    clf.fit(X, y)
    clf.fit(X_train, y_train)
    y_test_predicted = clf.predict(X_test)
    rmse.append(np.sqrt(mean_squared_error(y_test,y_test_predicted)))
    print('Lambda : %d | Test Error %.3f'%(lambd,rmse[-1]))


lambdas = np.logspace(-6, 10, 200)
clf = RidgeCV(alphas=lambdas).fit(X_train, y_train)
y_test_predicted = clf.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test,y_test_predicted))
print('Best Lambda in Ridge | Test Error %.3f'%(rmse))


lambdas = np.logspace(-6, 10, 200)
clf = LassoCV(alphas=lambdas).fit(X_train, y_train)
y_test_predicted = clf.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test,y_test_predicted))
print('Best Lambda in Lasso | Test Error %.3f'%(rmse))
