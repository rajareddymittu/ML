# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 09:15:11 2021

@author: Mr.BeHappy
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


cv=pd.read_csv('C:/Users/rajar/Documents/.summercoding/ML/to train/CORONA VIRUS/covid_19_india.csv')

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer

data=cv;
le=LabelEncoder();
data['State/UnionTerritory']=le.fit_transform(data['State/UnionTerritory'])


X=cv['Confirmed'].values.reshape(-1,1)
Y=cv['Deaths'].values.reshape(-1,1)
plt.scatter(X,Y);
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.3,random_state=0)

lin_reg=LinearRegression()
lin_reg.fit(X_train,y_train)
y_pred=lin_reg.predict(X_test)
plt.plot(X,y_pred,color='r',marker='o')
    
print("coeff:",lin_reg.coef_)
print("intercept:",lin_reg.intercept_)
from sklearn.metrics import r2_score
score=r2_score(y_pred,y_test)
print('prediction accuracy:',score)
import statsmodels.api as sm
X = sm.add_constant(X)
model= sm.OLS(Y, X).fit()
model.summary()








