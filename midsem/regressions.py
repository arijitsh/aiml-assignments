import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso, LassoCV
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings('ignore')

dataset = pd.read_csv('college.csv')

X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5,random_state=100)

reg = LinearRegression()
reg.fit(X_train, y_train)

y_test_predicted = reg.predict(X_test)

print("Coefficients by OLS\n\n")

for i in range(17):
    print("%s : %.2f"%(list(dataset)[i+1] ,reg.coef_[i]))

print("\n\nRMSE = % .2f"%np.sqrt(mean_squared_error(y_test,y_test_predicted)))

lambdas = [0,4,1e10]

for lambd in lambdas:
    reg = Ridge(alpha=lambd)
    reg.fit(X_train, y_train)
    y_test_predicted = reg.predict(X_test)

    print("\n\nCoefficients by Ridge with lambda = %d\n\n"%lambd)

    for i in range(17):
        print("%s : %.2f"%(list(dataset)[i+1] ,reg.coef_[i]))

    print("\n\nRMSE = % .2f"%np.sqrt(mean_squared_error(y_test,y_test_predicted)))

lambda_range = np.logspace(-6, 10, 20000)

reg = RidgeCV(alphas=lambda_range).fit(X_train, y_train)

print("\n\nCoefficients by Ridge with Best lambda (= %.2f)\n\n"%reg.alpha_)

reg = Ridge(alpha=reg.alpha_)
reg.fit(X_train, y_train)
y_test_predicted = reg.predict(X_test)


for i in range(17):
    print("%s : %.2f"%(list(dataset)[i+1] ,reg.coef_[i]))

print("\n\nRMSE = % .2f"%np.sqrt(mean_squared_error(y_test,y_test_predicted)))

for lambd in lambdas:
    reg = Lasso(alpha=lambd)
    reg.fit(X_train, y_train)
    y_test_predicted = reg.predict(X_test)

    print("\n\nCoefficients by Lasso with lambda = %d\n\n"%lambd)

    for i in range(17):
        print("%s : %.2f"%(list(dataset)[i+1] ,reg.coef_[i]))

    print("\n\nRMSE = % .2f"%np.sqrt(mean_squared_error(y_test,y_test_predicted)))

reg = LassoCV(alphas=lambda_range, cv=10, max_iter=100000).fit(X_train, y_train)

print("\n\nCoefficients by Lasso with Best lambda (= %.2f)\n\n"%reg.alpha_)

reg = Lasso(alpha=reg.alpha_)

reg.fit(X_train, y_train)
y_test_predicted = reg.predict(X_test)

for i in range(17):
    print("%s : %.2f"%(list(dataset)[i+1] ,reg.coef_[i]))

print("\n\nRMSE = % .2f"%np.sqrt(mean_squared_error(y_test,y_test_predicted)))
