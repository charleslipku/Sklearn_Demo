#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 8/13/2016 11:04 PM
# @Author  : ANG LI
# @Affiliation    : University of Arkansas
# @File    : GradientDescent.py
# @Software: PyCharm

import numpy as np
from sklearn.datasets import load_boston
from sklearn.linear_model import SGDRegressor
from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

data=load_boston()
X_Train, X_Test, Y_Train, Y_Test=train_test_split(data.data, data.target)
X_scaler=StandardScaler()
Y_scaler=StandardScaler()
X_Train=X_scaler.fit_transform(X_Train)
Y_Train=Y_scaler.fit_transform(Y_Train)
X_Test=X_scaler.fit_transform(X_Test)
Y_Test=Y_scaler.fit_transform(Y_Test)

model=SGDRegressor(loss='squared_loss')
cross_score=cross_val_score(model,X_Train,Y_Train,cv=5)

print 'Cross Score:', cross_score
print 'Avergae Score:', cross_score.mean()
print 'Test Score:', model.score(X_Test,Y_Test)