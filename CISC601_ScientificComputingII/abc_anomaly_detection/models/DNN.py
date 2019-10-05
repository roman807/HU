#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Roman Moser, 10/5/19

"""
Implementation of dense neural network (DNN)
"""

from keras.models import Sequential
from keras.layers import Dense

# input parameters:
EPOCHS = 10
BATCH_SIZE = 100
OPTIMIZER = 'adam'
ACTIVATION = 'tanh'
LOSS = 'binary_crossentropy'
VERBOSE = 1

class DNN:
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.input_dim = X_train.shape[1]
    
    def dnn(self, activation=ACTIVATION, optimizer=OPTIMIZER):
        model = Sequential()
        model.add(Dense(input_dim=self.input_dim, units=300, activation=ACTIVATION))
        model.add(Dense(units=100, activation=activation))
        model.add(Dense(units=1, activation='sigmoid'))
        model.compile(loss=LOSS, optimizer=optimizer)
        return model

    def fit(self):
        self.dnn = self.dnn()
        self.dnn.fit(
                x=self.X_train, 
                y=self.y_train, 
                batch_size=BATCH_SIZE, 
                epochs=EPOCHS, 
                verbose=VERBOSE
        )
    
    def predict(self, X_test):
        return self.dnn.predict(X_test)
