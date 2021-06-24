# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 11:17:39 2021

@author: Mr.BeHappy
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset=pd.read_csv("C:/Users/rajar/Documents/.summercoding/ML/to train/wearther/weather.csv")
dataset.plot(x='MinTemp',y='MaxTemp',style='o')
plt.title('min temp vs max temp')
plt.xlabel('MinTemp')
plt.ylabel('MaxTemp')
plt.plot()
plt.show()

X=dataset['MinTemp'].values.reshape(-1,1)
y=dataset['MaxTemp'].values.reshape(-1,1)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

regressor=LinearRegression()
regressor.fit(X_train,y_train)

print(regressor.intercept_)

print(regressor.coef_)
y_pred=regressor.predict(X_test)
df=pd.DataFrame({'Actual':y_test.flatten(),'predicted ':y_pred.flatten()})
print(df);
plt.scatter(X_test,y_test,color='gray')
plt.plot(X_test,y_pred,color='red',linewidth=2)
plt.show()

from sklearn.metrics import r2_score
r2=r2_score(y_test,y_pred)
print("r2 score ",r2)
print("the regression eqation : y=10.6619*x+0.92")





