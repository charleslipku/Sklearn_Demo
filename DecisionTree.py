#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 8/14/2016 11:43 PM
# @Author  : ANG LI
# @Affiliation    : University of Arkansas
# @File    : DecisionTree.py
# @Software: PyCharm

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier



if __name__ == '__main__':
    df=pd.read_csv('./dataset/ad.data', header=None)
    explanatory_variable_columns=set(df.columns.values)
    response_variable_column=df[len(df.columns.values)-1]
    explanatory_variable_columns.remove(len(df.columns.values)-1)

    x=df[list(explanatory_variable_columns)]
    y=[1 if e == 'ad.' else 0 for e in response_variable_column]

    x.replace(to_replace=' *\?', value=1, regex=True, inplace=True)
    x_train, x_test, y_train, y_test=train_test_split(x, y)

    #clf= DecisionTreeClassifier(criterion='entropy')
    clf=RandomForestClassifier(criterion='entropy')
    parameters={
        'n_estimators': [5, 10, 20, 50],
        'max_depth': [150, 155, 160],
        'min_samples_split': [1, 2, 3],
        'min_samples_leaf': [1, 2, 3],
    }

    grid_search=GridSearchCV(clf,parameters,n_jobs=-1, verbose=1, scoring='f1')
    grid_search.fit(x_train, y_train)
    best_params=grid_search.best_params_
    print 'Best score:', grid_search.best_score_
    print 'Best params:', best_params
    #print classification_report(y_test, grid_search.predict(x_test))