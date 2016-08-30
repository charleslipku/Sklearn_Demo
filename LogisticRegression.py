#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 8/14/2016 10:30 PM
# @Author  : ANG LI
# @Affiliation    : University of Arkansas
# @File    : LogisticRegression.py
# @Software: PyCharm

import pandas as pd
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.metrics.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV


def main():
    pipline=Pipeline(
        [
            ('vect', TfidfVectorizer(stop_words='english')),
            ('clf', LogisticRegression())
        ]
    )
    parameters={
        'vect_max_df': (0.25, 0.5, 0.75),
        'vect_ngram_range': ((1,1), (1,2)),
        'vect_use_idf': (True, False),
        'clf_C': (0.1, 1, 10),
    }
    df=pd.read_csv('./dataset/train.tsv', header=0, delimiter='\t')
    x, y=df['Phrase'], df['Sentiment'].as_matrix()
    x_train, x_test, y_train, y_test=train_test_split(x, y, train_size=0.5)
    grid_search=GridSearchCV(pipline,parameters,n_jobs=-1,verbose=1, scoring='accuracy')
    grid_search.fit(x_train, y_train)

    print 'Best score: %0.33f' % grid_search.best_score_
    # print 'Best parameters set:'
    # best_params=grid_search.best_estimator_.get_params()
    # for key in sorted(best_params.keys()):
    #     print '\t%s: %r' % (key, best_params[key])


if __name__=='__main__':
    main()
