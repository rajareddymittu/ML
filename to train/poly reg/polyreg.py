# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 07:42:51 2021

@author: Mr.BeHappy
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
data=pd.read_csv('C:/Users/rajar/Documents/.summercoding/ML/to train/poly reg/kc_house_data.csv')
X=data.drop(['price','date'],axis=1).values
Y=data.price.values
for i in range(1,20):
    poly=PolynomialFeatures(degree=i)
    X_train,X_test,Y_train,Y_test=train_test_split(poly.fit_transform(X),Y,test_size=0.2)
    from sklearn.metrics import r2_score
    from sklearn.linear_model import LinearRegression
    lin=LinearRegression()
    model=lin.fit(X_train,Y_train)
    r2=r2_score(Y_train,model.predict(X_train))
    print(i,"th r2 values is",r2)
















