#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Roman Moser, 6/19/19

"""
train various models on datasets with a variety of point difficulties.
Point difficulty is defined as how difficult it is for a Logistic Regression
to correctly label the point.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_curve, auc
import pickle
import time
from scipy import interp

import autoencoder

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

def results_point_difficulty(data_original, settings):
    #anom_freq=0.01, n_datasets=10):
    """Generate datasets with different point_difficulties of anomaly class.
       Train, predict and evaluate various models.
       Input:  * data_original: dict with all prepared datasets
               * anom_freq: relative frequency of anomalies (default: 1%)
               * n_datasets: number of datasets to be generated (default: 10)
       Output: * results_point_freq: dict with roc_auc score for each
                 generated dataset
    """
    results_dir = settings['results_dir']
    settings = settings['settings_point_difficulty']
    n_datasets = settings['n_datasets']
    results_point_difficulty_lr = dict()
    results_point_difficulty_gbm = dict()
    results_point_difficulty_iforest = dict()
    results_point_difficulty_lof = dict()
    results_point_difficulty_ae_unsupervised = dict()
    results_point_difficulty_ae_supervised = dict()
    
    for dataset in data_original.keys():
        print('train on dataset: {}'.format(dataset))
        results_point_difficulty_lr[dataset] = dict()
        results_point_difficulty_gbm[dataset] = dict()
        results_point_difficulty_iforest[dataset] = dict()
        results_point_difficulty_lof[dataset] = dict()
        results_point_difficulty_ae_unsupervised[dataset] = dict()
        results_point_difficulty_ae_supervised[dataset] = dict()
        data_reg = data_original[dataset]['regular']
        anom = data_original[dataset]['anom'].sort_values('point_difficulty')
        num_anom = np.round(settings['anom_freq'] * data_reg.shape[0] / \
                            (1 - settings['anom_freq']))
        step = np.round(anom.shape[0] / (n_datasets + 1))
        for i in range(n_datasets):
            y_pred_lr, y_pred_gbm, y_pred_iforest, y_pred_lof = [], [], [], []
            y_pred_ae_unsupervised, y_pred_ae_supervised, y_true = [], [], []
            #roc_auc_gbm, roc_auc_iforest, roc_auc_lof = [], [], []
            data_anom = anom.iloc[int(i * step) : int(min(i * step + num_anom, anom.shape[0])), :]
            data_sample = pd.concat([data_reg, data_anom]).sample(frac=1)\
                .reset_index(drop=True)
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
                results_point_difficulty_lr[dataset]\
                    [np.round(i / n_datasets, 2)] = (mean_fpr, mean_tpr, mean_auc)
            if settings['models_train']['gbm']:
                mean_fpr, mean_tpr, mean_auc = create_mean_roc_auc(y_true, y_pred_gbm)
                results_point_difficulty_gbm[dataset][np.round(i / n_datasets, 2)] = \
                    (mean_fpr, mean_tpr, mean_auc)
            if settings['models_train']['iforest']:
                mean_fpr, mean_tpr, mean_auc = create_mean_roc_auc(y_true, y_pred_iforest)
                results_point_difficulty_iforest[dataset]\
                    [np.round(i / n_datasets, 2)] = (mean_fpr, mean_tpr, mean_auc)
            if settings['models_train']['lof']:
                mean_fpr, mean_tpr, mean_auc = create_mean_roc_auc(y_true, y_pred_lof)
                results_point_difficulty_lof[dataset][np.round(i / n_datasets, 2)] = \
                    (mean_fpr, mean_tpr, mean_auc)                    
            if settings['models_train']['autoencoder_unsupervised']:
                mean_fpr, mean_tpr, mean_auc = create_mean_roc_auc(y_true, y_pred_ae_unsupervised)
                results_point_difficulty_ae_unsupervised[dataset]\
                    [np.round(i / n_datasets, 2)] = (mean_fpr, mean_tpr, mean_auc)
            if settings['models_train']['autoencoder_supervised']:
                mean_fpr, mean_tpr, mean_auc = create_mean_roc_auc(y_true, y_pred_ae_supervised)
                results_point_difficulty_ae_supervised[dataset]\
                    [np.round(i / n_datasets, 2)] = (mean_fpr, mean_tpr, mean_auc)

    timestr = time.strftime("%H%M%S")
    if settings['models_train']['lr']:
        name = 'results_point_difficulty_lr_{}'.format(timestr)
        save_results(results_point_difficulty_lr, results_dir, name)
    if settings['models_train']['gbm']:
        name = 'results_point_difficulty_gbm_{}'.format(timestr)
        save_results(results_point_difficulty_gbm, results_dir, name)    
    if settings['models_train']['iforest']:
        name = 'results_point_difficulty_iforest_{}'.format(timestr)
        save_results(results_point_difficulty_iforest, results_dir, name)      
    if settings['models_train']['lof']:
        name = 'results_point_difficulty_lof_{}'.format(timestr)
        save_results(results_point_difficulty_lof, results_dir, name)        
    if settings['models_train']['autoencoder_unsupervised']:
        name = 'results_point_difficulty_ae_unsupervised_{}'.format(timestr)
        save_results(results_point_difficulty_ae_unsupervised, results_dir, name) 
    if settings['models_train']['autoencoder_supervised']:
        name = 'results_point_difficulty_ae_supervised_{}'.format(timestr)
        save_results(results_point_difficulty_ae_supervised, results_dir, name) 
        
def plot_results_point_difficulty(data_original, results_point_difficulty, model_names, settings):
    """Plot results for all original dataset over spectrum of generated
       datasets with various point difficulties
       Input:  * results_point_difficulty: dict with roc_auc score for each
                 generated dataset
    """
    plots_dir = settings['plots_dir']
    y = dict()
    for model_name in model_names:
        y[model_name] = []
    for dataset in data_original.keys():
        colors = ['blue', 'green', 'red', 'orange', 'purple', 'darkblue', 'black']\
            [:len(model_names)]
        x = np.array(list(results_point_difficulty[0][dataset].keys()))
        for model_name, results, color in zip(model_names, results_point_difficulty, colors):
            y_ = [i[2] for i in results[dataset].values()]
            y[model_name].append(y_)
            plt.plot(x, np.array(y_), color=color)
        plt.legend(model_names)
        plt.title('Results point difficulty {}'.format(dataset))
        plt.xlabel('point difficulty')
        plt.ylabel('ROC AUC')
        plt.savefig(plots_dir + '/results_point_difficulty_{}.png'.format(dataset))
        plt.clf()
    for model_name, color in zip(model_names, colors):
        plt.plot(x, np.mean(np.array(y[model_name]), axis=0), color=color)
    plt.legend(model_names)
    plt.title('Results point difficulty - mean')
    plt.xlabel('point difficulty')
    plt.ylabel('ROC AUC')
    plt.savefig(plots_dir + '/results_point_difficulty_mean.png')
    plt.clf()


