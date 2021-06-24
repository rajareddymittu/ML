# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 11:18:42 2021

@author: Mr.BeHappy
"""

import pandas as pd;
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder

companies=pd.read_csv('C:/Users/rajar/Documents/.summercoding/ML/to train/profit prediction/1000_companies.csv')
data=companies
companies.head()
x=companies['R&D Spend'].values.reshape(-1,1)
y=companies['State'].values.reshape(-1,1)
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
le=LabelEncoder()
data.State=le.fit_transform(data.State)
columnTransformer = ColumnTransformer([('encoder',OneHotEncoder(),[3])],remainder='passthrough')
  
data = np.array(columnTransformer.fit_transform(data), dtype = np.float64)

#extracting features
X=data[:,:-1]
#extracting targets
Y=data[:,-1]
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.3,random_state=0)

lin_reg=LinearRegression() 
lin_reg.fit(X_train,y_train)
y_pred=lin_reg.predict(X_test)
print("coeff:",lin_reg.coef_)
print("intercept:",lin_reg.intercept_)
from sklearn.metrics import r2_score
score=r2_score(y_pred,y_test)
print('prediction accuracy:',score)
import statsmodels.api as sm
X = sm.add_constant(X)
model= sm.OLS(Y, X).fit()
model.summary()
