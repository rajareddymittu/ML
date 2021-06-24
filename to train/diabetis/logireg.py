# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 11:34:48 2021

@author: Mr.BeHappy
"""

import numpy as np

import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import metrics

import seaborn as sns
data= pd.read_csv('C:/Users/rajar/Documents/.summercoding/ML/to train/diabetis/diabetes.csv')
dd=pd.DataFrame({ 'hi':data['Pregnancies']},index=data['Glucose'])
dd.describe()
X=data.iloc[:,:-1]
y=data.iloc[:,-1]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)
logreg=LogisticRegression()
logreg.fit(X_train,y_train)
y_pred=logreg.predict(X_test)
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(df)
conf_matrix=metrics.confusion_matrix(y_test,y_pred)
print(conf_matrix)
sns.heatmap(conf_matrix,annot=True)
plt.title('confusion matrix')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
print("Accuracy:", metrics.accuracy_score(y_test,y_pred))





