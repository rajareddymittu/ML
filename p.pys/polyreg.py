# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 10:07:25 2021

@author: Mr.BeHappy
"""

import numpy as np
import matplotlib.pyplot as plt

# X=np.array([80,40,-40,-120,-200,-280])
# y=np.array([6.47,6.24,5.72,5.09,4.3,3.33])
X=np.array([5,20,40,60,80,100])
y=np.array([0.0002,0.0012,0.0060,0.0300,0.0600,0.1200])
X=X.reshape(-1,1)
y=y.reshape(-1,1)

from sklearn.linear_model import LinearRegression
lin = LinearRegression() 
lin.fit(X, y)
max=0.0;

for i  in range(15):
    from sklearn.preprocessing import PolynomialFeatures
    
    poly = PolynomialFeatures(degree = i)
    X_poly = poly.fit_transform(X)
      
    lin2 = LinearRegression()
    lin2.fit(X_poly, y)
    
    plt.scatter(X, y, color = 'blue')
      
    plt.plot(X, lin.predict(X), color = 'red')
    plt.title('Linear Regression')
    plt.xlabel('Temperature')
    plt.ylabel('Pressure')
      
    plt.show()
    
    plt.scatter(X, y, color = 'blue')
      
    plt.plot(X, lin2.predict(poly.fit_transform(X)), color = 'red')
    plt.title('Polynomial Regression')
    plt.xlabel('Temperature')
    plt.ylabel('Pressure')
      
    plt.show()
    from sklearn.metrics import r2_score
    r2=r2_score(y,lin2.predict(poly.fit_transform(X)))
    
    
    print (i,"th degreee  r2 score is " ,r2)















