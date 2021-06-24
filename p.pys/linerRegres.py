# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 11:17:39 2021

@author: Mr.BeHappy
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklrearn.metrics import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
dataset=pd.read_csv("weather.csv")
