#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Roman Moser, 7/14/19

"""
Run all models on one data set, show ROC curve
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt
import os

os.chdir('/home/roman/Documents/HU/CISC600_ScientificComputing/Project/')

import autoencoder

def main(): 
    # Prepare data
    df = pd.read_csv('data/creditcard.csv')
    df = df.drop(['Time'], axis=1)
    sc = StandardScaler()
    df.iloc[:, :-1] = sc.fit_transform(df.iloc[:, :-1])
    
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]                
    skf = StratifiedKFold(n_splits=3, random_state=123)
    
    fpr, tpr, roc_auc = dict(), dict(), dict()
    
    # Logistic Regression
    y_true, y_pred_proba = [], []
    for train_index, test_index in skf.split(X, y):
        clf = LogisticRegression()
        clf.fit(X.iloc[train_index, :], y[train_index])
        y_true.append(y[test_index])
        y_pred_proba.append(clf.predict_proba(X.iloc[test_index, :])[:, 1])
    y_pred_proba = np.concatenate(y_pred_proba)
    y_true = np.concatenate(y_true)
    fpr['lr'], tpr['lr'], _ = roc_curve(y_true, y_pred_proba)
    roc_auc['lr'] = np.round(roc_auc_score(y_true, y_pred_proba),3)
    
    # GBM
    y_true, y_pred_proba = [], []
    for train_index, test_index in skf.split(X, y):
        clf = GradientBoostingClassifier()
        clf.fit(X.iloc[train_index, :], y[train_index])
        y_true.append(y[test_index])
        y_pred_proba.append(clf.predict_proba(X.iloc[test_index, :])[:, 1])
    y_pred_proba = np.concatenate(y_pred_proba)
    y_true = np.concatenate(y_true)
    fpr['gbm'], tpr['gbm'], _ = roc_curve(y_true, y_pred_proba)
    roc_auc['gbm'] = np.round(roc_auc_score(y_true, y_pred_proba), 3)
    
    # LOF
    y_true, y_pred_proba = [], []
    for train_index, test_index in skf.split(X, y):
        X_train_unsupervised = X.iloc[train_index, :][y[train_index]==0]
        clf = LocalOutlierFactor()
        clf.fit(X_train_unsupervised)
        decision_function = clf.decision_function(X.iloc[test_index, :])
        y_true.append(y[test_index])
        y_pred_proba.append(1 - np.interp(decision_function, \
                                    (decision_function.min(), 
                                     decision_function.max()), (0, 1)))
    y_pred_proba = np.concatenate(y_pred_proba)
    y_true = np.concatenate(y_true)
    fpr['lof'], tpr['lof'], _ = roc_curve(y_true, y_pred_proba)
    roc_auc['lof'] = np.round(roc_auc_score(y_true, y_pred_proba), 3)
    
    # Isolation Forest
    y_true, y_pred_proba = [], []
    for train_index, test_index in skf.split(X, y):
        X_train_unsupervised = X.iloc[train_index, :][y[train_index]==0]
        clf = IsolationForest()
        clf.fit(X_train_unsupervised)
        decision_function = clf.decision_function(X.iloc[test_index, :])
        y_true.append(y[test_index])
        y_pred_proba.append(1 - np.interp(decision_function, \
                                    (decision_function.min(), 
                                     decision_function.max()), (0, 1)))
    y_pred_proba = np.concatenate(y_pred_proba)
    y_true = np.concatenate(y_true)
    fpr['if'], tpr['if'], _ = roc_curve(y_true, y_pred_proba)
    roc_auc['if'] = np.round(roc_auc_score(y_true, y_pred_proba))
    
    # Autoencoder unsupervised
    y_true, y_pred_proba = [], []
    for train_index, test_index in skf.split(X, y):
        X_train_unsupervised = X.iloc[train_index, :][y[train_index]==0]
        input_dim = X_train_unsupervised.shape[1]
        clf = autoencoder.autoencoder_unsupervised(input_dim=input_dim)
        clf.fit(X_train_unsupervised, X_train_unsupervised, 
               batch_size=50, epochs=3, verbose=1)
        y_true.append(y[test_index])
        X_test_pred = clf.predict(X.iloc[test_index, :])
        y_pred_proba.append(autoencoder.\
                        reconstruction_error(X.iloc[test_index, :], X_test_pred))
    y_pred_proba = np.concatenate(y_pred_proba)
    y_true = np.concatenate(y_true)
    fpr['ae_unsupervised'], tpr['ae_unsupervised'], _ = roc_curve(y_true, y_pred_proba)
    roc_auc['ae_unsupervised'] = np.round(roc_auc_score(y_true, y_pred_proba), 3)
    
    # Autoencoder supervised
    y_true, y_pred_proba = [], []
    for train_index, test_index in skf.split(X, y):
        input_dim = X.iloc[train_index, :].shape[1]
        y_train = pd.concat([X.iloc[train_index, :], y[train_index]], axis=1)
        clf = autoencoder.autoencoder_supervised(input_dim=input_dim)
        clf.fit(X.iloc[train_index, :], y_train, batch_size=50, epochs=3, verbose=0)
        y_true.append(y[test_index])
        X_test_pred = clf.predict(X.iloc[test_index, :])
        y_pred_proba.append(autoencoder.\
                        reconstruction_error(X.iloc[test_index, :], X_test_pred))
    y_pred_proba = np.concatenate(y_pred_proba)
    y_true = np.concatenate(y_true)
    fpr['ae_supervised'], tpr['ae_supervised'], _ = roc_curve(y_true, y_pred_proba)
    roc_auc['ae_supervised'] = np.round(roc_auc_score(y_true, y_pred_proba), 3)
    
    # Plot ROC AUC curves
    colors = ['blue', 'green', 'red', 'orange', 'purple', 'darkblue', 'black']
    for i, model in enumerate(fpr.keys()):
        print(str(model))
        print(roc_auc[model])
        plt.plot(fpr[model], tpr[model], color=colors[i])
    plt.legend(roc_auc.items())
    plt.title('ROC AUC curves all models')
    plt.show()
    plt.savefig('results/plots/roc_curves_all_models.png')

if __name__ == '__main__':
    main()
