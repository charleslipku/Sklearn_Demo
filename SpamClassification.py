#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 8/14/2016 2:55 PM
# @Author  : ANG LI
# @Affiliation    : University of Arkansas
# @File    : SpamClassification.py
# @Software: PyCharm

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from  sklearn.linear_model.logistic import LogisticRegression
from sklearn.cross_validation import cross_val_score, train_test_split

df=pd.read_csv('./dataset/SMSSpamCollection', delimiter='\t', header=None)
x_train_raw, x_test_raw, y_train_raw, y_test_raw=train_test_split(df[1], df[0])
vectorizer=TfidfVectorizer()
x_train=vectorizer.fit_transform(x_train_raw)
x_test=vectorizer.transform(x_test_raw)

model=LogisticRegression()
model.fit(x_train, y_train_raw)

predictions=model.predict(x_test)
print predictions[:5]
print x_test_raw[:5]
