# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 10:40:27 2021

@author: Mr.BeHappy
"""

import pandas as pd;
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder

companies=pd.read_csv('C:/Users/rajar/Documents/.summercoding/ML/to train/profit prediction/1000_companiess.csv')
data=companies
companies.head()
# x=companies['R&D Spend'].values.reshape(-1,1)
# y=companies['profit'].values.reshape(-1,1)
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
  

#extracting features
X=companies.drop(columns='Profit').values
#extracting targets
Y=data['Profit'].values.reshape(-1,1)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.2)
   
for i in range(1,25):
    from sklearn.preprocessing import PolynomialFeatures    
    poly = PolynomialFeatures(degree = i)
    X_poly = poly.fit_transform(X)
    lin2 = LinearRegression()
    lin2.fit(X_poly, Y)
    
          
    plt.show()
    from sklearn.metrics import r2_score
    r2=r2_score(Y,lin2.predict(poly.fit_transform(X)))
    print("r2 score for ",i,"th degree is ",r2)
    
