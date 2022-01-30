# -*- coding: utf-8 -*-
"""
Created on Sun Aug 22 14:12:14 2021

@author: Rownok
"""

import pandas as pd 

dataset=pd.read_csv('Job Satisfaction survey.csv')


x = dataset.iloc[:, [0, 2, 3, 4]].values
y = dataset.iloc[:, -1].values


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

# from sklearn.linear_model import LogisticRegression
# classifier=LogisticRegression(random_state=0)
# classifier.fit(x_train,y_train)

from sklearn.svm import SVC
classifier=SVC(kernel='poly', random_state=0)
classifier.fit(x_train,y_train)


predict=classifier.predict(x_test)


from sklearn.metrics import confusion_matrix
finalresult=confusion_matrix(y_test,predict)
