# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 18:40:14 2022

@author: helder.santos
"""
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
import pandas as pd

train = pd.read_csv('C:/Users/helder.santos/Desktop/Helder/MADSAD/DM_I/training.csv')
test = pd.read_csv('C:/Users/helder.santos/Desktop/Helder/MADSAD/DM_I/task.csv')

train.describe()
test.describe()

for coln in train.columns:
    if train[coln].isnull().values.any():
        print(coln)
        
train.fillna(value=train['x9'].mean(), inplace=True)
test.fillna(value=train['x9'].mean(), inplace=True)

resid = test[['i']]
train = train.drop(columns='i')
test = test.drop(columns='i')

X_train = train.iloc[:,1:]
y_train = train.iloc[:,0]
X_test = test.iloc[:,1:]

clf = RandomForestClassifier(n_estimators=100)
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

resid['y']=y_pred

resid.to_csv('C:/Users/helder.santos/Desktop/Helder/MADSAD/DM_I/randomforest.csv',index=False)

print("Accuracy:",metrics.accuracy_score(y_train, y_pred))

# import matplotlib.pyplot as plt
# plt.matshow(train.corr())
# plt.title('Correlation Plot')
# plt.show()