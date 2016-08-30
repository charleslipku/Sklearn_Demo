#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 8/13/2016 12:26 AM
# @Author  : ANG LI
# @Affiliation    : University of Arkansas
# @File    : MultipleLinearRegression.py
# @Software: PyCharm
from sklearn.linear_model import LinearRegression

X = [[6, 2], [8, 1], [10, 0], [14, 2], [18, 0]]
Y = [[7], [9], [13], [17.5], [18]]
X_Test = [[8, 2], [9, 0], [11, 2], [16, 2], [12, 0]]
Y_Test = [[11], [8.5], [15], [18], [11]]

model=LinearRegression()
model.fit(X,Y)
predictions=model.predict(X_Test)
for i, prediction in enumerate(predictions):
    print 'Predicted: %s, Target: %s' % (prediction,Y_Test[i])

score=model.score(X_Test,Y_Test)
print 'Model Score:', score