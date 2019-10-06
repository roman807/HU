#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Roman Moser, 10/5/19

import numpy as np
from sklearn.ensemble import IsolationForest

class IForest:
    """
    Implementation of isolation forest (wrapper for IsolationForest from sklearn)
    """
    def __init__(self, X_train, y_train):
        self.X_train = X_train[y_train==0]
        self.y_train = y_train[y_train==0]
        self.clf = IsolationForest()
        
    def fit(self):
        self.clf.fit(self.X_train)
    
    def predict(self, X):
        decision_function = self.clf.decision_function(X)
        return 1 - np.interp(decision_function, (decision_function.min(), \
                                             decision_function.max()), (0, 1))
