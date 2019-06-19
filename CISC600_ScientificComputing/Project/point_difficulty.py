#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Roman Moser, 6/19/19


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, KFold


def point_difficulty(df, kfold_splits=10):
    """Calulate point difficulty according to prediction accuracy with 
       Logistic Regression
       * Input: data frame, kfold_splits (default: 10)
       * Output: Point difficulty: value between 0 and 1
    """
    point_difficulty = np.zeros(shape=df.shape[0])
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    kf = KFold(n_splits=kfold_splits)
    kf.get_n_splits(X)    
    for train_index, test_index in kf.split(X, y):
        lr = LogisticRegression()
        lr.fit(X.loc[train_index, :], y[train_index])
        y_proba = lr.predict_proba(X.loc[test_index, :])
        point_difficulty[test_index] = y_proba[:, 1] - y[test_index]
    return abs(point_difficulty)


def results_point_difficulty(data_original, anom_freq=0.01, n_datasets=10):
    """Generate datasets with different point_difficulties of anomaly class.
       Train, predict and evaluate various models.
       Input:  * data_original: dict with all prepared datasets
               * anom_freq: relative frequency of anomalies (default: 1%)
               * n_datasets: number of datasets to be generated (default: 10)
       Output: * results_point_freq: dict with roc_auc score for each
                 generated dataset
    """
    results_point_difficulty = dict()
    for dataset in data_original.keys():
        results_point_difficulty[dataset] = dict()
        results_point_difficulty[dataset]['gbm'] = dict()
        results_point_difficulty[dataset]['iforest'] = dict()
        results_point_difficulty[dataset]['lof'] = dict()
        data_reg = data_original[dataset]['regular']
        anom = data_original[dataset]['anom'].sort_values('point_difficulty')
        num_anom = np.round(anom_freq * data_reg.shape[0] / (1 - anom_freq))
        step = np.round(anom.shape[0] / (n_datasets + 1))
        for i in range(n_datasets):
            roc_auc_gbm, roc_auc_iforest, roc_auc_lof = [], [], []
            data_anom = anom.iloc[int(i * step) : int(min(i * step + num_anom, anom.shape[0])), :]
            data_sample = pd.concat([data_reg, data_anom]).sample(frac=1)\
                .reset_index(drop=True)
            X = data_sample.iloc[:, :-2]
            y = data_sample.iloc[:, -2]                
            skf = StratifiedKFold(n_splits=3)
            for train_index, test_index in skf.split(X, y):
                X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
                y_train, y_test = y[train_index], y[test_index]
                # GBM:
                gbm =  GradientBoostingClassifier()
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
            
            results_point_difficulty[dataset]['gbm'][np.round(i / n_datasets, 2)] = \
                np.mean(roc_auc_gbm)
            results_point_difficulty[dataset]['iforest'][np.round(i / n_datasets, 2)] = \
                np.mean(roc_auc_iforest)
            results_point_difficulty[dataset]['lof'][np.round(i / n_datasets, 2)] = \
                np.mean(roc_auc_lof)
    return results_point_difficulty


# Plot ROC AUC:
def plot_results_point_difficulty(data_original, results_point_difficulty):
    """Plot results for all original dataset over spectrum of generated
       datasets with various point difficulties
       Input:  * results_point_difficulty: dict with roc_auc score for each
                 generated dataset
    """
    roc_auc_gbm, roc_auc_iforest, roc_auc_lof = [], [], []
    for dataset in data_original.keys():
        x = np.array(list(results_point_difficulty[dataset]['gbm'].keys()))
        y_gbm = np.array(list(results_point_difficulty[dataset]['gbm'].values()))
        y_iforest = np.array(list(results_point_difficulty[dataset]['iforest'].values()))
        y_lof = np.array(list(results_point_difficulty[dataset]['lof'].values()))
        roc_auc_gbm.append(y_gbm)
        roc_auc_iforest.append(y_iforest)
        roc_auc_lof.append(y_lof)
        plt.plot(x, y_gbm, color='green')
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
