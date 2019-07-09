#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Roman Moser, 6/19/19

"""
train various models on datasets with a variety of relative frequencies
of anomalies. Range of anomaly frequencies specified in inputs.py
"""

import pandas as pd
import numpy as np
import pickle
import time
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from scipy import interp

import autoencoder

def save_results(obj, folder_name, name):
    with open('results/'+ folder_name + '/' + name + '.pkl', 'wb') as f:
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


def results_relative_frequency(data_original, settings):
    """Generate datasets with different ratios ofanomalies:total samples 
       (relative frequency) from a set of original datasets. Train, predict
        and evaluate various models.
       Input:  * data_original: dict with all prepared datasets
               * anom_freq: array with range of relative frequencies
       Output: * results_relative_freq: dict with roc_auc score for each
                 generated dataset
    """
    results_dir = settings['results_dir']
    settings = settings['settings_relative_frequency']
    seed = 123
    n_samples = settings['n_random_samples']
    results_relative_freq_lr = dict()
    results_relative_freq_gbm = dict()
    results_relative_freq_iforest = dict()
    results_relative_freq_lof = dict()
    results_relative_freq_ae_unsupervised = dict()
    results_relative_freq_ae_supervised = dict()
    for dataset in data_original.keys():
        print('train on dataset: {}'.format(dataset))
        results_relative_freq_lr[dataset] = dict()
        results_relative_freq_gbm[dataset] = dict()
        results_relative_freq_iforest[dataset] = dict()
        results_relative_freq_lof[dataset] = dict()
        results_relative_freq_ae_unsupervised[dataset] = dict()
        results_relative_freq_ae_supervised[dataset] = dict()
        data_reg = data_original[dataset]['regular']
        num_anom = np.round(settings['anom_freq'] * data_reg.shape[0] / 
                            (1 - settings['anom_freq']))
        for i, num in enumerate(num_anom):
            y_pred_lr, y_pred_gbm, y_pred_iforest, y_pred_lof = [], [], [], []
            y_pred_ae_unsupervised, y_pred_ae_supervised, y_true = [], [], []
            # n different random samples to reduce variance
            for s in range(settings['n_random_samples']): 
                data_anom = data_original[dataset]['anom'].\
                        sample(n=int(num), random_state=seed)
                data_sample = pd.concat([data_reg, data_anom]).\
                        sample(frac=1, random_state=seed).reset_index(drop=True)
                seed += 1
                X = data_sample.iloc[:, :-2]
                y = data_sample.iloc[:, -2]                
                skf = StratifiedKFold(n_splits=3)
                for train_index, test_index in skf.split(X, y):
                    X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
                    y_train, y_test = y[train_index], y[test_index]
                    X_train_unsupervised = X_train[y_train==0]
                    y_true.append(y_test)
                    
                    # Logistic Regression
                    if settings['models_train']['lr']:
                        lr = LogisticRegression()
                        lr.fit(X_train, y_train)
                        y_pred_lr.append(lr.predict_proba(X_test)[:, 1])                    
                    
                    # GBM
                    if settings['models_train']['gbm']:
                        gbm = GradientBoostingClassifier()
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
                results_relative_freq_lr[dataset]\
                    [np.round(settings['anom_freq'][i], 3)] = (mean_fpr, mean_tpr, mean_auc)
            if settings['models_train']['gbm']:
                mean_fpr, mean_tpr, mean_auc = create_mean_roc_auc(y_true, y_pred_gbm)
                results_relative_freq_gbm[dataset][np.round(settings['anom_freq'][i], 3)] = \
                    (mean_fpr, mean_tpr, mean_auc)
            if settings['models_train']['iforest']:
                mean_fpr, mean_tpr, mean_auc = create_mean_roc_auc(y_true, y_pred_iforest)
                results_relative_freq_iforest[dataset]\
                    [np.round(settings['anom_freq'][i], 3)] = (mean_fpr, mean_tpr, mean_auc)
            if settings['models_train']['lof']:
                mean_fpr, mean_tpr, mean_auc = create_mean_roc_auc(y_true, y_pred_lof)
                results_relative_freq_lof[dataset][np.round(settings['anom_freq'][i], 3)] = \
                    (mean_fpr, mean_tpr, mean_auc)                    
            if settings['models_train']['autoencoder_unsupervised']:
                mean_fpr, mean_tpr, mean_auc = create_mean_roc_auc(y_true, y_pred_ae_unsupervised)
                results_relative_freq_ae_unsupervised[dataset]\
                    [np.round(settings['anom_freq'][i], 3)] = (mean_fpr, mean_tpr, mean_auc)
            if settings['models_train']['autoencoder_supervised']:
                mean_fpr, mean_tpr, mean_auc = create_mean_roc_auc(y_true, y_pred_ae_supervised)
                results_relative_freq_ae_supervised[dataset]\
                    [np.round(settings['anom_freq'][i], 3)] = (mean_fpr, mean_tpr, mean_auc)
                    
    timestr = time.strftime("%H%M%S")
    if settings['models_train']['lr']:
        name = 'results_relative_frequency_lr_{}_{}'.format(n_samples, timestr)
        save_results(results_relative_freq_lr, results_dir, name)
    if settings['models_train']['gbm']:
        name = 'results_relative_frequency_gbm_{}_{}'.format(n_samples, timestr)
        save_results(results_relative_freq_gbm, results_dir, name)    
    if settings['models_train']['iforest']:
        name = 'results_relative_frequency_iforest_{}_{}'.format(n_samples, timestr)
        save_results(results_relative_freq_iforest, results_dir, name)      
    if settings['models_train']['lof']:
        name = 'results_relative_frequency_lof_{}_{}'.format(n_samples, timestr)
        save_results(results_relative_freq_lof, results_dir, name)        
    if settings['models_train']['autoencoder_unsupervised']:
        name = 'results_relative_frequency_ae_unsupervised_{}_{}'.format(n_samples, timestr)
        save_results(results_relative_freq_ae_unsupervised, results_dir, name) 
    if settings['models_train']['autoencoder_supervised']:
        name = 'results_relative_frequency_ae_supervised_{}_{}'.format(n_samples, timestr)
        save_results(results_relative_freq_ae_supervised, results_dir, name) 


def plot_results_relative_frequency(data_original, results_relative_freq, model_names, settings):
    """Plot results for all original dataset over spectrum of generated
       datasets with various relative frequencies
       Input:  * results_relative_freq: dict with roc_auc score for each
                 generated dataset
    """
    plots_dir = settings['plots_dir']
    y = dict()
    for model_name in model_names:
        y[model_name] = []
    for dataset in data_original.keys():
        colors = ['blue', 'green', 'red', 'orange', 'purple', 'darkblue', 'black']\
            [:len(model_names)]
        x = np.array(list(results_relative_freq[0][dataset].keys()))
        for model_name, results, color in zip(model_names, results_relative_freq, colors):
            y_ = [i[2] for i in results[dataset].values()]
            y[model_name].append(y_)
            plt.plot(x, np.array(y_), color=color)
        plt.legend(model_names)
        plt.title('Results relative frequency {}'.format(dataset))
        plt.xlabel('relative frequency')
        plt.ylabel('ROC AUC')
        plt.savefig(plots_dir + 'results_relative_frequency_{}.png'.format(dataset))
        plt.clf()
    for model_name, color in zip(model_names, colors):
        plt.plot(x, np.mean(np.array(y[model_name]), axis=0), color=color)
    plt.legend(model_names)
    plt.title('Results relative frequency - mean')
    plt.xlabel('relative frequency')
    plt.ylabel('ROC AUC')
    plt.savefig(plots_dir + '/results_relative_frequency_mean.png')
    plt.clf()
    
