#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 8/13/2016 10:19 PM
# @Author  : ANG LI
# @Affiliation    : University of Arkansas
# @File    : RegularizedLR.py
# @Software: PyCharm

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score

df = pd.read_csv('./dataset/winequality-red.csv', sep=';')
# print df.describe()

# plt.scatter(df['alcohol'], df['quality'])
# plt.xlabel('Alcohol')
# plt.ylabel('Quality')
# plt.title('Alcohol Against Quality')
# plt.show()

X = df[list(df.columns)[:-1]]
Y = df['quality']
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y)

model = LinearRegression()
model.fit(X_Train, Y_Train)
Y_Predictions = model.predict(X_Test)
model_score = model.score(X_Test, Y_Test)
cross_score = cross_val_score(model, X, Y, cv=10)
print Y_Predictions
print 'Model Score:', model_score
print 'Corss Score:', cross_score.mean(),cross_score
