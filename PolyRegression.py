#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 8/13/2016 12:40 AM
# @Author  : ANG LI
# @Affiliation    : University of Arkansas
# @File    : PolyRegression.py
# @Software: PyCharm

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

X = [[6], [8], [10], [14], [18]]
Y = [[7], [9], [13], [17.5], [18]]
X_Test=[[8],[9],[11],[16],[12]]
Y_Test=[[11],[8.5],[15],[18],[11]]

regressor=LinearRegression()
regressor.fit(X,Y)
xx=np.linspace(0,26,100)
yy=regressor.predict(xx.reshape(xx.shape[0],1))
plt.plot(xx,yy)

quadratic_featurizer=PolynomialFeatures(degree=2)
X_Train_quadratic=quadratic_featurizer.fit_transform(X)
X_Test_quadratic=quadratic_featurizer.fit_transform(X_Test)

quadratic_regressor=LinearRegression()
quadratic_regressor.fit(X_Train_quadratic,Y)

xx_quadratic=quadratic_featurizer.transform(xx.reshape(xx.shape[0],1))
plt.plot(xx, quadratic_regressor.predict(xx_quadratic),c='r',linestyle='--')
plt.grid(True)
plt.scatter(X,Y)
plt.show()

score=quadratic_regressor.score(X_Test_quadratic,Y_Test)
print 'Model Score:', score