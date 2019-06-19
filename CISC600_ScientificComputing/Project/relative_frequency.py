#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Roman Moser, 6/19/19


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


def results_relative_frequency(data_original, anom_freq):
    """Generate datasets with different ratios ofanomalies:total samples 
       (relative frequency) from a set of original datasets. Train, predict
        and evaluate various models.
       Input:  * data_original: dict with all prepared datasets
               * anom_freq: array with range of relative frequencies
       Output: * results_relative_freq: dict with roc_auc score for each
                 generated dataset
    """
    results_relative_freq = dict()
    for dataset in data_original.keys():
        results_relative_freq[dataset] = dict()
        results_relative_freq[dataset]['gbm'] = dict()
        results_relative_freq[dataset]['iforest'] = dict()
        results_relative_freq[dataset]['lof'] = dict()
        data_reg = data_original[dataset]['regular']
        num_anom = np.round(anom_freq * data_reg.shape[0] / (1 - anom_freq))
        for i, num in enumerate(num_anom):
            roc_auc_gbm, roc_auc_iforest, roc_auc_lof = [], [], []
            for s in range(10):   # 10 different random samples to reduce variance
                data_anom = data_original[dataset]['anom'].sample(n=int(num))
                data_sample = pd.concat([data_reg, data_anom]).sample(frac=1)\
                    .reset_index(drop=True)
                X = data_sample.iloc[:, :-2]
                y = data_sample.iloc[:, -2]                
                skf = StratifiedKFold(n_splits=3)
                for train_index, test_index in skf.split(X, y):
                    X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
                    y_train, y_test = y[train_index], y[test_index]
                    # GBM
                    gbm = GradientBoostingClassifier()
                    gbm.fit(X_train, y_train)
                    y_proba_gbm = gbm.predict_proba(X_test)
                    roc_auc_gbm.append(roc_auc_score(y_test, y_proba_gbm[:, 1]))                    
                    # Isolation Forest:
                    X_train_unsupervised = X_train[y_train==0]
                    iforest = IsolationForest()
                    iforest.fit(X_train_unsupervised)
                    decision_function = iforest.decision_function(X_test)
                    y_proba_if = 1 - np.interp(decision_function, \
                                                      (decision_function.min(), 
                                                       decision_function.max()), (0, 1))
                    roc_auc_iforest.append(roc_auc_score(y_test, y_proba_if))
                    # Local Outlier Factor (LOF):
                    lof = LocalOutlierFactor()
                    lof.fit(X_train_unsupervised)
                    decision_function = lof._decision_function(X_test)
                    y_proba_lof = 1 - np.interp(decision_function, \
                                                      (decision_function.min(), 
                                                       decision_function.max()), (0, 1))
                    roc_auc_lof.append(roc_auc_score(y_test, y_proba_lof))                
            results_relative_freq[dataset]['gbm'][np.round(anom_freq[i], 3)] = \
                np.mean(roc_auc_gbm)
            results_relative_freq[dataset]['iforest'][np.round(anom_freq[i], 3)] = \
                np.mean(roc_auc_iforest)
            results_relative_freq[dataset]['lof'][np.round(anom_freq[i], 3)] = \
                np.mean(roc_auc_lof)
    return results_relative_freq


def plot_results_relative_frequency(data_original, results_relative_freq):
    """Plot results for all original dataset over spectrum of generated
       datasets with various relative frequencies
       Input:  * results_relative_freq: dict with roc_auc score for each
                 generated dataset
    """
    roc_auc_gbm, roc_auc_iforest, roc_auc_lof = [], [], []
    for dataset in data_original.keys():
        x = np.array(list(results_relative_freq[dataset]['gbm'].keys()))
        y_gbm = np.array(list(results_relative_freq[dataset]['gbm'].values()))
        y_iforest = np.array(list(results_relative_freq[dataset]['iforest'].values()))
        y_lof = np.array(list(results_relative_freq[dataset]['lof'].values()))
        roc_auc_gbm.append(y_gbm)
        roc_auc_iforest.append(y_iforest)
        roc_auc_lof.append(y_lof)
        plt.plot(x, y_gbm, color='blue')
        plt.plot(x, y_iforest, color='red')
        plt.plot(x, y_lof, color='orange')
        plt.title('ROC AUC - {}'.format(dataset))
        plt.show()
    
    mean_roc_auc_gbm = np.mean(np.array(roc_auc_gbm), axis=0)
    plt.plot(x, mean_roc_auc_gbm, color='blue')
    mean_roc_auc_iforest = np.mean(np.array(roc_auc_iforest), axis=0)
    plt.plot(x, mean_roc_auc_iforest, color='red')
    mean_roc_auc_lof = np.mean(np.array(roc_auc_lof), axis=0)
    plt.plot(x, mean_roc_auc_lof, color='orange')
    plt.title('ROC AUC - mean over all data sets'.format(dataset))
    plt.show()
