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


def variance(X):
    """Calculate semantic variance of anomaly data (Input dataframe X).
       Semantic variance is defined as the mean of the eigenvalues of
       the covariance matrix
       Input:  * X: Dataframe of anomaly data
       Output: * semantic variance
    """
    var_covar = np.cov(X, rowvar=False)
    eigenvalues = np.linalg.eig(var_covar)[0]
    return np.mean(eigenvalues)

def results_semanitc_variation(data_original, anom_freq=0.01, n_datasets=10):
    """Generate datasets with different semantic variation of anomaly class.
       Higher semantic variation means anomalies are spread out further.
       Train, predict and evaluate various models.
       Input:  * data_original: dict with all prepared datasets
               * anom_freq: relative frequency of anomalies (default: 1%)
               * n_datasets: number of datasets to be generated (default: 10)
       Output: * results_semanitc_variation: dict with roc_auc score for each
                 generated dataset
    """
    results_semantic_variation = dict()
    for dataset in data_original.keys():
        results_semantic_variation[dataset] = dict()
        results_semantic_variation[dataset]['gbm'] = dict()
        results_semantic_variation[dataset]['iforest'] = dict()
        results_semantic_variation[dataset]['lof'] = dict()
        data_reg = data_original[dataset]['regular']
        anom = data_original[dataset]['anom']
        # calculate mean euclidean distance from three random points (non-outliers):
        ref_points = data_reg.iloc[np.random.choice(data_reg.shape[0], 3, replace=False), :]
        euc_dist = []
        for i in range(anom.shape[0]):
            euc_dist_ref = []
            for j in range(ref_points.shape[0]):
                euc_dist_ref.append(np.linalg.norm(anom.iloc[i, :-2] - ref_points.iloc[j, :-2]))
            euc_dist.append(np.mean(euc_dist_ref))
        anom['euc_dist'] = euc_dist
        anom.sort_values('euc_dist', inplace=True)
        # create datasets according to euclidean distance:
        num_anom = np.round(anom_freq * data_reg.shape[0] / (1 - anom_freq))
        step = np.round(anom.shape[0] / (n_datasets + 1))
        datasets_anom, var = [], []
        for i in range(n_datasets):
            datasets_anom.append(anom.iloc[int(i * step) : int(min(i * step + \
                                           num_anom, anom.shape[0])), :])
            var.append(variance(datasets_anom[-1].iloc[:, :-3]))
        var = [v.real for v in var]
        # sort datasets according to variance:
        datasets_anom = [dataset for _, dataset in sorted(zip(var, datasets_anom))]
        for i, dataset_anom in enumerate(datasets_anom):
            roc_auc_gbm, roc_auc_iforest, roc_auc_lof = [], [], []
            data_sample = pd.concat([data_reg, dataset_anom.iloc[:, :-1]]).\
                sample(frac=1).reset_index(drop=True)
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
            
            results_semantic_variation[dataset]['gbm'][np.round(i / n_datasets, 2)] = \
                np.mean(roc_auc_gbm)
            results_semantic_variation[dataset]['iforest'][np.round(i / n_datasets, 2)] = \
                np.mean(roc_auc_iforest)
            results_semantic_variation[dataset]['lof'][np.round(i / n_datasets, 2)] = \
                np.mean(roc_auc_lof)
    return results_semantic_variation
    

def plot_results_semanitc_variation(data_original, results_semanitc_variation):
    """Plot results for all original dataset over spectrum of generated
       datasets with various semanitc variations
       Input:  * results_semanitc_variation: dict with roc_auc score for each
                 generated dataset
    """                                
    roc_auc_gbm, roc_auc_iforest, roc_auc_lof = [], [], []
    for dataset in data_original.keys():
        x = np.array(list(results_semanitc_variation[dataset]['gbm'].keys()))
        y_gbm = np.array(list(results_semanitc_variation[dataset]['gbm'].values()))
        y_iforest = np.array(list(results_semanitc_variation[dataset]['iforest'].values()))
        y_lof = np.array(list(results_semanitc_variation[dataset]['lof'].values()))
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
    