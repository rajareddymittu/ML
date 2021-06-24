# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 11:58:17 2021

@author: Mr.BeHappy
"""

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
digits_data = load_digits()
digits = digits_data.data
targets = digits_data.target
digits.shape
targets.shape
a_digit = np.split(digits[0], 8)
plt.imshow(a_digit, cmap='gray')
x_train, x_test, y_train, y_test = train_test_split(digits, targets, test_size=0.25)
print(x_train.shape, x_test.shape)
from sklearn.linear_model import LogisticRegression
logistic_reg = LogisticRegression()
logistic_reg.fit(x_train, y_train)
predict_data = x_test[23].reshape(1, -1)
plt.imshow(np.split(x_test[23], 8), cmap='gray')
a=logistic_reg.predict(x_test[23].reshape(1, -1))
print('The identified digit is:',a)