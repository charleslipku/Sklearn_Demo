#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 8/12/2016 11:57 PM
# @Author  : ANG LI
# @Affiliation    : University of Arkansas
# @File    : SimpleLinearRegression.py
# @Software: PyCharm

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

X = [[6], [8], [10], [14], [18]]
Y = [[7], [9], [13], [17.5], [18]]
X_Test=[[8],[9],[11],[16],[12]]
Y_Test=[[11],[8.5],[15],[18],[11]]
plt.figure()
plt.title('Pizza Price')
plt.xlabel('Diameter in inches')
plt.ylabel('Price in dollars')
plt.plot(X, Y, 'k.')
plt.grid(True)
# plt.show()

model = LinearRegression()
model.fit(X, Y)

loss = np.mean((model.predict(X) - Y) ** 2)
score=model.score(X_Test,Y_Test)
print 'Loss:', loss
print 'Model Score:', score
print model.predict(12)[0]
