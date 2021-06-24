# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 10:47:34 2021

@author: Mr.BeHappy
"""

import numpy as np
import matplotlib.pyplot as plt

x=np.array([95,85,80,70,60])
y=np.array([85,95,70,65,70])
#x is independent and y is dependent variable, 
# we are storing those data in the array
 
n=np.size(x);
m_x,m_y=np.mean(x),np.mean(y)#mean
ss_xy=np.sum(x*y)-n*(m_x*m_y)#
ss_xx=np.sum(x*x)-n*m_x*m_x

b0_1=ss_xy/ss_xx# slope
b0_0=m_y-b0_1*m_x#intercept

np.append(x,70)#to predict the value of 70,
np.append(y,b0_0+b0_1*70)#predicted value
y_pred=b0_0+b0_1*x#list of predicted values
print("intercept:",b0_0)
print("slope:    ",b0_1)
plt.scatter(x,y)#plotting the actual values on graph
plt.plot(x,y_pred,color='r',marker='o')#plotting the predicted values on graph
from sklearn.metrics import r2_score
r2=r2_score(y,y_pred)#greater the score(coefficient of determination) better the prediction

print("r2 =",r2)
r=r2**0.5
print ("r=",r)


