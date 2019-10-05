#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Roman Moser, 10/5/19

"""
Implementation of unsupervised autoencoder according to Yuki Yamanaka, Tomoharu
Iwata, Hiroshi Takahashi, Masanori Yamada, Sekitoshi Kanai 
https://arxiv.org/pdf/1903.10709.pdf
"""

import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# input parameters:
EPOCHS = 10
BATCH_SIZE = 100
OPTIMIZER = 'adam'
ACTIVATION = 'tanh'
LOSS = 'mean_squared_error'
VERBOSE = 1

class AE:
    def __init__(self, X_train, y_train):
        self.X_train = X_train[y_train==0]
        self.input_dim = X_train.shape[1]

    def reconstruction_error(self, X_true, X_pred):
        diff_squared = np.sqrt(((X_true - X_pred) ** 2).sum(axis=1))
        error = 1 - np.exp(-diff_squared / diff_squared.mean())
        return error
    
    def abc(self, activation=ACTIVATION, optimizer=OPTIMIZER):
        """
        Note: input to supervised auto_encoder: 
        y_true = pd.concat([X_train, y_train])
        because X_train and y_train are both required in custom loss function
        """
        model = Sequential()
        model.add(Dense(input_dim=self.input_dim, units=300, activation=ACTIVATION))
        model.add(Dense(units=100, activation=activation))
        model.add(Dense(units=100, activation=activation))
        model.add(Dense(units=300, activation=activation))
        model.add(Dense(units=self.input_dim, activation=activation))
        model.compile(loss=LOSS, optimizer=optimizer)
        return model

    def fit(self):
        self.abc = self.abc()
        self.abc.fit(
                self.X_train, 
                self.X_train, 
                batch_size=BATCH_SIZE, 
                epochs=EPOCHS, 
                verbose=VERBOSE
        )
    
    def predict(self, X_test):
        X_test_pred = self.abc.predict(X_test)
        return self.reconstruction_error(X_test, X_test_pred)
