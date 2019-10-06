#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Roman Moser, 10/5/19

"""
Implementation of dense neural network (DNN)
"""

from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

# input parameters:
EPOCHS = 300
BATCH_SIZE = 100
OPTIMIZER = 'adam'
ACTIVATION = 'tanh'
LOSS = 'binary_crossentropy'
VERBOSE = 1

class DNN:
    def __init__(self, X, y):
        self.X_train, self.X_val, self.y_train, self.y_val = \
            train_test_split(X, y, test_size=0.2)
        self.input_dim = self.X_train.shape[1]
    
    def dnn(self, activation=ACTIVATION, optimizer=OPTIMIZER):
        model = Sequential()
        model.add(Dense(input_dim=self.input_dim, units=300, activation=ACTIVATION))
        model.add(Dense(units=100, activation=activation))
        model.add(Dense(units=1, activation='sigmoid'))
        model.compile(loss=LOSS, optimizer=optimizer)
        return model

    def fit(self):
        filepath = 'models/trained/dnn_weights.best.hdf5'
        checkpoints = [ModelCheckpoint(
                filepath, 
                monitor='val_loss', 
                verbose=1, 
                save_best_only=True, 
                mode='min'
        )]
        self.dnn = self.dnn()
        self.dnn.fit(
                x=self.X_train, 
                y=self.y_train,
                validation_data=(self.X_val, self.y_val),
                callbacks=checkpoints,
                batch_size=BATCH_SIZE, 
                epochs=EPOCHS, 
                verbose=VERBOSE
        )
    
    def predict(self, X_test):
        self.dnn.load_weights('models/trained/dnn_weights.best.hdf5')
        return self.dnn.predict(X_test)
