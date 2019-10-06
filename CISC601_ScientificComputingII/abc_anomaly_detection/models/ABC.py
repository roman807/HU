#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Roman Moser, 10/5/19

"""
Implementation of autoencoder according to Yuki Yamanaka, Tomoharu Iwata,
Hiroshi Takahashi, Masanori Yamada, Sekitoshi Kanai 
https://arxiv.org/pdf/1903.10709.pdf
"""

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
import keras.backend as K
from sklearn.model_selection import train_test_split

# input parameters:
EPOCHS = 100
BATCH_SIZE = 100
OPTIMIZER = 'adam'
ACTIVATION = 'tanh'
VERBOSE = 1

class ABC:
    def __init__(self, X, y):
        self.X = X
        self.y = pd.concat([X, y], axis=1)
        self.X_train, self.X_val, self.y_train, self.y_val = \
            train_test_split(self.X, self.y, test_size=0.2)
        self.input_dim = self.X_train.shape[1]

    def reconstruction_error(self, X_true, X_pred):
        diff_squared = np.sqrt(((X_true - X_pred) ** 2).sum(axis=1))
        error = 1 - np.exp(-diff_squared / diff_squared.mean())
        return error

    def custom_loss(self, y_true, X_pred):
        """
        y=0: normal point, y=1: anomaly
        """
        X = y_true[:, :-1]
        y = y_true[:, -1]
        l2_norm = K.sqrt(K.sum(K.square(X - X_pred), axis=1))
        return (1 - y) * l2_norm - y * l2_norm

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
        model.compile(loss=self.custom_loss, optimizer=optimizer)
        return model

    def fit(self):
        filepath = 'models/trained/abc_weights.best.hdf5'
        checkpoints = [ModelCheckpoint(
                filepath, 
                monitor='val_loss', 
                verbose=1, 
                save_best_only=True, 
                mode='min'
        )]
        self.abc = self.abc()
        self.abc.fit(
                self.X_train,
                self.y_train,
                validation_data=(self.X_val, self.y_val),
                callbacks=checkpoints,
                batch_size=BATCH_SIZE, 
                epochs=EPOCHS, 
                verbose=VERBOSE
        )
    
    def predict(self, X_test):
        self.abc.load_weights('models/trained/abc_weights.best.hdf5')
        X_test_pred = self.abc.predict(X_test)
        return self.reconstruction_error(X_test, X_test_pred)
