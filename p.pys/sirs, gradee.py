# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 11:35:54 2021

@author: Mr.BeHappy
"""

import numpy as np
import matplotlib.pyplot as plt
x=np.array([1,2,3,4,5])
y=np.array([5,7,9,11,13])
m_curr=float(0)
b_curr=float(0)
lr=0.02#learning_rate
n=len(x)
itr=200#no of iterations
plt.scatter(x,y)
cost=[]  #difference of acutal and predicted
for i in range(itr):#interating  itr times
    y_pred=m_curr*x+b_curr
    #prediction in every interation
    cost_tmp=(1/n)*sum([val**2 for val in (y-y_pred)])
    #calculationg difference
    cost.append(cost_tmp)
    
    dm=-(2/n)*sum(x*(y-y_pred))
    db=-(2/n)*sum(y-y_pred)
    m_curr=m_curr-lr*dm
    b_curr=b_curr-lr*db
    
    #print("m {},b {}, cost {}, iteration {}".format(m_curr,b_curr,cost_tmp,i))
plt.plot(x,y_pred)
    
from sklearn.metrics import r2_score
r2=r2_score(y, y_pred)
print("r2 =",r2)
plt.show()
plt.figure
index=np.arange(200)
plt.scatter(index,cost)
plt.show