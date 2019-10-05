#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Roman Moser, 10/5/19

from sklearn.linear_model import LogisticRegression

class LogReg:
    """
    Implementation of logistic regression (wrapper for LogisticRegression from sklearn)
    """
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.clf = LogisticRegression()
        
    def fit(self):
        self.clf.fit(self.X_train, self.y_train)
    
    def predict(self, X):
        return self.clf.predict_proba(X)[:, 1]
