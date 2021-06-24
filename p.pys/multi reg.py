# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 13:17:28 2021

@author: Mr.BeHappy
"""

import pandas as pd;
import numpy as np
from sklearn import linear_model
df=pd.read_csv('C:/Users/rajar/Documents/.summercoding/ML/to train/profit prediction/1000_Companies.csv')
reg=linear_model.LinearRegression()
reg.fit(df[['R&D Spend','Administration','Marketing Spend']],df.Profit)





