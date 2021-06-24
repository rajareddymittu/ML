
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 10:47:34 2021

@author: Mr.BeHappy
"""

import numpy as np
import matplotlib.pyplot as plt
x=np.array([95,85,80,70,60])
y=np.array([684,984,6815326,81641,78498])


n=np.size(x);
m_x,m_y=np.mean(x),np.mean(y)
ss_xy=np.sum(x*y)-n*(m_x*m_y)
ss_xx=np.sum(x*x)-n*m_x*m_x

b0_1=ss_xy/ss_xx
b0_0=m_y-b0_1*m_x
np.append(x,70)
np.append(y,b0_0+b0_1*70)
y_pred=b0_0+b0_1*x
print("intercept:",b0_0)
print("slope:    ",b0_1)
plt.scatter(x,y)
plt.plot(x,y_pred,color='r',marker='o')
from sklearn.metrics import r2_score
r2=r2_score(y,y_pred)
print("r2 =",r2)
r=r2**0.5
print ("r=",r)


