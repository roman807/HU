#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Roman Moser, 6/19/19

"""
Implementation of autoencoder according to Yuki Yamanaka, Tomoharu Iwata,
Hiroshi Takahashi, Masanori Yamada, Sekitoshi Kanai 
https://arxiv.org/pdf/1903.10709.pdf
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from keras.models import Sequential
from keras.layers import Dense
import keras.backend as K


def reconstruction_error(X_true, X_pred):
    diff_squared = np.sqrt(((X_true - X_pred) ** 2).sum(axis=1))
    error = 1 - np.exp(-diff_squared)
    return error

def custom_loss(y_true, X_pred):
    X = y_true[:, :-1]
    y = y_true[:, -1]
    l2_norm = K.sqrt(K.sum(K.square(X - X_pred), axis=1))
    return y * K.log(1 - K.exp(-l2_norm)) + (1 - y) * l2_norm

def autoencoder_unsupervised(input_dim, activation='tanh', 
                             loss='mean_squared_error', optimizer='adam'):
    model = Sequential()
    model.add(Dense(input_dim=input_dim, units=20, activation=activation))
    model.add(Dense(units=5, activation=activation))
    model.add(Dense(units=20, activation=activation))
    model.add(Dense(units=input_dim, activation=activation))
    model.compile(loss=loss, optimizer=optimizer)
    return model

def autoencoder_supervised(input_dim, activation='tanh', 
                           loss=custom_loss, optimizer='adam'):
    model = Sequential()
    model.add(Dense(input_dim=input_dim, units=20, activation=activation))
    model.add(Dense(units=5, activation=activation))
    model.add(Dense(units=20, activation=activation))
    model.add(Dense(units=input_dim, activation=activation))
    model.compile(loss=loss, optimizer=optimizer)
    return model





