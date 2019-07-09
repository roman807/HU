#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Roman Moser, 6/19/19

"""
train various models on datasets with a variety of semantic variances.
Semantic variance refers to the variance among anomalies. To get semantic
variance of a dataset, calculate the mean of the eigenvalues of the 
covariance matrix.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
import pickle
import time
from scipy import interp

import autoencoder

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

def save_results(obj, folder_name, name):
    with open(folder_name + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f)
        
def create_mean_roc_auc(y_true_l, y_pred_l):
    tprs_l = []
    mean_fpr = np.linspace(0, 1, 10000)
    for y_true, y_pred in zip(y_true_l, y_pred_l):
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        tprs_l.append(interp(mean_fpr, fpr, tpr))
        tprs_l[-1][0] = 0.0
    mean_tpr = np.mean(tprs_l, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    return mean_fpr, mean_tpr, mean_auc

def results_semanitc_variation(data_original, settings):
    """Generate datasets with different semantic variation of anomaly class.
       Higher semantic variation means anomalies are spread out further.
       Train, predict and evaluate various models.
       Input:  * data_original: dict with all prepared datasets
               * anom_freq: relative frequency of anomalies (default: 1%)
               * n_datasets: number of datasets to be generated (default: 10)
       Output: * results_semanitc_variation: dict with roc_auc score for each
                 generated dataset
    """
    results_dir = settings['results_dir']
    settings = settings['settings_semantic_variance']
    n_datasets = settings['n_datasets']
    results_semantic_variance_lr = dict()
    results_semantic_variance_gbm = dict()
    results_semantic_variance_iforest = dict()
    results_semantic_variance_lof = dict()
    results_semantic_variance_ae_unsupervised = dict()
    results_semantic_variance_ae_supervised = dict()
    for dataset in data_original.keys():
        print('train on dataset: {}'.format(dataset))
        results_semantic_variance_lr[dataset] = dict()
        results_semantic_variance_gbm[dataset] = dict()
        results_semantic_variance_iforest[dataset] = dict()
        results_semantic_variance_lof[dataset] = dict()
        results_semantic_variance_ae_unsupervised[dataset] = dict()
        results_semantic_variance_ae_supervised[dataset] = dict()
        data_reg = data_original[dataset]['regular']
        anom = data_original[dataset]['anom']
        
        # calculate mean euclidean distance from three random points (non-outliers):
        ref_points = data_reg.iloc[np.random.choice(data_reg.shape[0], 3, \
                                            replace=False), :]
        euc_dist = []
        for i in range(anom.shape[0]):
            euc_dist_ref = []
            for j in range(ref_points.shape[0]):
                euc_dist_ref.append(np.linalg.norm(anom.iloc[i, :-2] - ref_points.iloc[j, :-2]))
            euc_dist.append(np.mean(euc_dist_ref))
        anom['euc_dist'] = euc_dist
        anom.sort_values('euc_dist', inplace=True)
        # create datasets according to euclidean distance:
        num_anom = np.round(settings['anom_freq'] * data_reg.shape[0] / \
                            (1 - settings['anom_freq']))
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
            y_pred_lr, y_pred_gbm, y_pred_iforest, y_pred_lof = [], [], [], []
            y_pred_ae_unsupervised, y_pred_ae_supervised, y_true = [], [], []
            data_sample = pd.concat([data_reg, dataset_anom.iloc[:, :-1]]).\
                sample(frac=1).reset_index(drop=True)
            X = data_sample.iloc[:, :-2]
            y = data_sample.iloc[:, -2]                
            skf = StratifiedKFold(n_splits=3)
            for train_index, test_index in skf.split(X, y):
                X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
                y_train, y_test = y[train_index], y[test_index]  
                X_train_unsupervised = X_train[y_train==0]
                y_true.append(y_test)
                
                # Logistic Regression:
                if settings['models_train']['lr']:
                    lr = LogisticRegression()
                    lr.fit(X_train, y_train)
                    y_pred_lr.append(lr.predict_proba(X_test)[:, 1])
                    
                # GBM:
                if settings['models_train']['gbm']:
                    gbm =  GradientBoostingClassifier()
                    gbm.fit(X_train, y_train)
                    y_pred_gbm.append(gbm.predict_proba(X_test)[:, 1])

                # Isolation Forest:
                if settings['models_train']['iforest']:
                    iforest = IsolationForest()
                    iforest.fit(X_train_unsupervised)
                    decision_function = iforest.decision_function(X_test)
                    y_pred_iforest.append(1 - np.interp(decision_function, \
                                                (decision_function.min(), 
                                                 decision_function.max()), (0, 1)))
                
                # Local Outlier Factor (LOF):
                if settings['models_train']['lof']:
                    lof = LocalOutlierFactor()
                    lof.fit(X_train_unsupervised)
                    decision_function = lof._decision_function(X_test)
                    y_pred_lof.append(1 - np.interp(decision_function, \
                                                (decision_function.min(), 
                                                 decision_function.max()), (0, 1)))
                    
                # Autoencoder unsupervised
                if settings['models_train']['autoencoder_unsupervised']:
                    input_dim = X_train_unsupervised.shape[1]
                    ae = autoencoder.autoencoder_unsupervised(input_dim=input_dim)
                    ae.fit(X_train_unsupervised, X_train_unsupervised, 
                           batch_size=50, epochs=2, verbose=0)
                    X_test_pred = ae.predict(X_test)
                    y_pred_ae_unsupervised.append(autoencoder.\
                                    reconstruction_error(X_test, X_test_pred))

                # Autoencoder supervised
                if settings['models_train']['autoencoder_supervised']:
                    input_dim = X_train.shape[1]
                    ae = autoencoder.autoencoder_supervised(input_dim=input_dim)
                    y_train = pd.concat([X_train, y_train], axis=1)
                    ae.fit(X_train, y_train, batch_size=50, epochs=2, verbose=0)
                    X_test_pred = ae.predict(X_test)
                    y_pred_ae_supervised.append(autoencoder.\
                                    reconstruction_error(X_test, X_test_pred))  
                    
            if settings['models_train']['lr']:
                mean_fpr, mean_tpr, mean_auc = create_mean_roc_auc(y_true, y_pred_lr)
                results_semantic_variance_lr[dataset]\
                    [np.round(i / n_datasets, 2)] = (mean_fpr, mean_tpr, mean_auc)
            if settings['models_train']['gbm']:
                mean_fpr, mean_tpr, mean_auc = create_mean_roc_auc(y_true, y_pred_gbm)
                results_semantic_variance_gbm[dataset][np.round(i / n_datasets, 2)] = \
                    (mean_fpr, mean_tpr, mean_auc)
            if settings['models_train']['iforest']:
                mean_fpr, mean_tpr, mean_auc = create_mean_roc_auc(y_true, y_pred_iforest)
                results_semantic_variance_iforest[dataset]\
                    [np.round(i / n_datasets, 2)] = (mean_fpr, mean_tpr, mean_auc)
            if settings['models_train']['lof']:
                mean_fpr, mean_tpr, mean_auc = create_mean_roc_auc(y_true, y_pred_lof)
                results_semantic_variance_lof[dataset][np.round(i / n_datasets, 2)] = \
                    (mean_fpr, mean_tpr, mean_auc)                    
            if settings['models_train']['autoencoder_unsupervised']:
                mean_fpr, mean_tpr, mean_auc = create_mean_roc_auc(y_true, y_pred_ae_unsupervised)
                results_semantic_variance_ae_unsupervised[dataset]\
                    [np.round(i / n_datasets, 2)] = (mean_fpr, mean_tpr, mean_auc)
            if settings['models_train']['autoencoder_supervised']:
                mean_fpr, mean_tpr, mean_auc = create_mean_roc_auc(y_true, y_pred_ae_supervised)
                results_semantic_variance_ae_supervised[dataset]\
                    [np.round(i / n_datasets, 2)] = (mean_fpr, mean_tpr, mean_auc)
                    
    timestr = time.strftime("%H%M%S")
    if settings['models_train']['lr']:
        name = 'results_semantic_variance_lr_{}'.format(timestr)
        save_results(results_semantic_variance_lr, results_dir, name)
    if settings['models_train']['gbm']:
        name = 'results_semantic_variance_gbm_{}'.format(timestr)
        save_results(results_semantic_variance_gbm, results_dir, name)    
    if settings['models_train']['iforest']:
        name = 'results_semantic_variance_iforest_{}'.format(timestr)
        save_results(results_semantic_variance_iforest, results_dir, name)      
    if settings['models_train']['lof']:
        name = 'results_semantic_variance_lof_{}'.format(timestr)
        save_results(results_semantic_variance_lof, results_dir, name)        
    if settings['models_train']['autoencoder_unsupervised']:
        name = 'results_semantic_variance_ae_unsupervised_{}'.format(timestr)
        save_results(results_semantic_variance_ae_unsupervised, results_dir, name) 
    if settings['models_train']['autoencoder_supervised']:
        name = 'results_semantic_variance_ae_supervised_{}'.format(timestr)
        save_results(results_semantic_variance_ae_supervised, results_dir, name) 
    
# Plot ROC AUC:
def plot_results_semantic_variance(data_original, results_semantic_variance, model_names, settings):
    """Plot results for all original dataset over spectrum of generated
       datasets with various semantic variances
       Input:  * results_semantic_variance: dict with roc_auc score for each
                 generated dataset
    """
    plots_dir = settings['plots_dir']
    y = dict()
    for model_name in model_names:
        y[model_name] = []
    for dataset in data_original.keys():
        colors = ['blue', 'green', 'red', 'orange', 'purple', 'darkblue', 'black']\
            [:len(model_names)]
        x = np.array(list(results_semantic_variance[0][dataset].keys()))
        for model_name, results, color in zip(model_names, results_semantic_variance, colors):
            y_ = [i[2] for i in results[dataset].values()]
            y[model_name].append(y_)
            plt.plot(x, np.array(y_), color=color)
        plt.legend(model_names)
        plt.title('Results semantic variance {}'.format(dataset))
        plt.xlabel('semantic variance')
        plt.ylabel('ROC AUC')
        plt.savefig(plots_dir + '/results_semantic_variance_{}.png'.format(dataset))
        plt.clf()
    for model_name, color in zip(model_names, colors):
        plt.plot(x, np.mean(np.array(y[model_name]), axis=0), color=color)
    plt.legend(model_names)
    plt.title('Results semantic variance - mean')
    plt.xlabel('semantic variance')
    plt.ylabel('ROC AUC')
    plt.savefig(plots_dir + '/results_semantic_variance_mean.png')
    plt.clf()
