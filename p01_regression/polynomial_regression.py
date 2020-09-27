import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline

test_rmse = []
train_rmse = []
dataset = pd.read_csv('winequality-red.csv', sep=';')

X = dataset.iloc[:, 0:-1].values
y = dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)


for deg in range(1,6):
    poly_reg = make_pipeline(
        PolynomialFeatures(degree=deg),
        LinearRegression()
        )
    X_poly = poly_reg.fit(X_train, y_train)
    y_train_predicted = poly_reg.predict(X_train)
    y_test_predicted = poly_reg.predict(X_test)
    train_rmse.append(np.sqrt(mean_squared_error(y_train,y_train_predicted)))
    test_rmse.append(np.sqrt(mean_squared_error(y_test,y_test_predicted)))
    print('Degree : %d Training Error %.3f Test Error %.3f'%(deg,train_rmse[-1],test_rmse[-1]))

deg = list(range(1,6))

plt.scatter(deg, train_rmse)
plt.title('Training Error varying with degree')
plt.xlabel('Degree')
plt.ylabel('Root Mean Squared Error')
plt.show()

plt.scatter(deg, test_rmse)
plt.title('Testing Error varying with degree')
plt.xlabel('Degree')
plt.ylabel('Root Mean Squared Error')
plt.show()
