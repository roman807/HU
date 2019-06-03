#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Roman Moser, 6/2/19

"""
Quora Insincere Questions - prepare data with word2vec
run with: python3 main.py 
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler
import pickle
import glob
import time

import os
os.chdir('/home/roman/Documents/HU/CISC600_ScientificComputing/Project')

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

##### Semantic Variation: ADD DOCSTRING
def variance(X):
    var_covar = np.cov(X, rowvar=False)
    eigenvalues = np.linalg.eig(var_covar)[0]
    return np.mean(eigenvalues)

def save_results(obj, name):
    with open('results/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f)

def load_results(name):
    with open(name, 'rb') as f:
        return pickle.load(f)

##### Relative Frequency
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
        results_relative_freq[dataset]['lr'] = dict()
        results_relative_freq[dataset]['rf'] = dict()
        results_relative_freq[dataset]['gbm'] = dict()
        results_relative_freq[dataset]['iforest'] = dict()
        results_relative_freq[dataset]['lof'] = dict()
        data_reg = data_original[dataset]['regular']
        num_anom = np.round(anom_freq * data_reg.shape[0] / (1 - anom_freq))
        for i, num in enumerate(num_anom):
            roc_auc_lr, roc_auc_rf, roc_auc_gbm, roc_auc_iforest, roc_auc_lof = \
                [], [], [], [], []
            for s in range(5):   # 10 different random samples to reduce variance
                data_anom = data_original[dataset]['anom'].sample(n=int(num))
                data_sample = pd.concat([data_reg, data_anom]).sample(frac=1)\
                    .reset_index(drop=True)
                X = data_sample.iloc[:, :-2]
                y = data_sample.iloc[:, -2]                
                skf = StratifiedKFold(n_splits=3)
                for train_index, test_index in skf.split(X, y):
                    # Logistic Regression:
                    X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
                    y_train, y_test = y[train_index], y[test_index]
                    lr = LogisticRegression()
                    lr.fit(X_train, y_train)
                    y_proba = lr.predict_proba(X_test)
                    roc_auc_lr.append(roc_auc_score(y_test, y_proba[:, 1]))
                    # Random Forests:
                    rf =  RandomForestClassifier()
                    rf.fit(X_train, y_train)
                    y_proba_rf = rf.predict_proba(X_test)
                    roc_auc_rf.append(roc_auc_score(y_test, y_proba_rf[:, 1]))
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
            results_relative_freq[dataset]['lr'][np.round(anom_freq[i], 3)] = \
                np.mean(roc_auc_lr)
            results_relative_freq[dataset]['rf'][np.round(anom_freq[i], 3)] = \
                np.mean(roc_auc_rf)
            results_relative_freq[dataset]['gbm'][np.round(anom_freq[i], 3)] = \
                np.mean(roc_auc_gbm)
            results_relative_freq[dataset]['iforest'][np.round(anom_freq[i], 3)] = \
                np.mean(roc_auc_iforest)
            results_relative_freq[dataset]['lof'][np.round(anom_freq[i], 3)] = \
                np.mean(roc_auc_lof)
    timestr = time.strftime("%H%M%S")
    name = 'results_relative_freq_{}'.format(timestr)
    save_results(results_relative_freq, name)
    #return name

# Plot ROC AUC: 
def plot_results_relative_frequency(data_original, results_relative_freq):
    """Plot results for all original dataset over spectrum of generated
       datasets with various relative frequencies
       Input:  * results_relative_freq: dict with roc_auc score for each
                 generated dataset
    """
    roc_auc_lr, roc_auc_rf, roc_auc_gbm, roc_auc_iforest, roc_auc_lof = [], [], [], [], []
    for dataset in data_original.keys():
        x = np.array(list(results_relative_freq[dataset]['lr'].keys()))
        y_lr = np.array(list(results_relative_freq[dataset]['lr'].values()))
        y_rf = np.array(list(results_relative_freq[dataset]['rf'].values()))
        y_gbm = np.array(list(results_relative_freq[dataset]['gbm'].values()))
        y_iforest = np.array(list(results_relative_freq[dataset]['iforest'].values()))
        y_lof = np.array(list(results_relative_freq[dataset]['lof'].values()))
        roc_auc_lr.append(y_lr)
        roc_auc_rf.append(y_rf)
        roc_auc_gbm.append(y_gbm)
        roc_auc_iforest.append(y_iforest)
        roc_auc_lof.append(y_lof)
        plt.plot(x, y_lr, color='blue')
        plt.plot(x, y_rf, color='lightblue')
        plt.plot(x, y_gbm, color='green')
        plt.plot(x, y_iforest, color='red')
        plt.plot(x, y_lof, color='orange')
        plt.title('ROC AUC - {}'.format(dataset))
    plt.show()
    
    mean_roc_auc_lr = np.mean(np.array(roc_auc_lr), axis=0)
    plt.plot(x, mean_roc_auc_lr, color='blue')
    mean_roc_auc_rf = np.mean(np.array(roc_auc_rf), axis=0)
    plt.plot(x, mean_roc_auc_rf, color='lightblue')
    mean_roc_auc_gbm = np.mean(np.array(roc_auc_gbm), axis=0)
    plt.plot(x, mean_roc_auc_gbm, color='green')
    mean_roc_auc_iforest = np.mean(np.array(roc_auc_iforest), axis=0)
    plt.plot(x, mean_roc_auc_iforest, color='red')
    mean_roc_auc_lof = np.mean(np.array(roc_auc_lof), axis=0)
    plt.plot(x, mean_roc_auc_lof, color='orange')
    plt.title('ROC AUC - mean over all data sets'.format(dataset))
    plt.show()


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
        results_point_difficulty[dataset]['lr'] = dict()
        results_point_difficulty[dataset]['rf'] = dict()
        results_point_difficulty[dataset]['gbm'] = dict()
        results_point_difficulty[dataset]['iforest'] = dict()
        results_point_difficulty[dataset]['lof'] = dict()
        data_reg = data_original[dataset]['regular']
        anom = data_original[dataset]['anom'].sort_values('point_difficulty')
        num_anom = np.round(anom_freq * data_reg.shape[0] / (1 - anom_freq))
        step = np.round(anom.shape[0] / (n_datasets + 1))
        for i in range(n_datasets):
            roc_auc_lr, roc_auc_rf, roc_auc_gbm, roc_auc_iforest, roc_auc_lof = \
                [], [], [], [], []
            data_anom = anom.iloc[int(i * step) : int(min(i * step + num_anom, anom.shape[0])), :]
            data_sample = pd.concat([data_reg, data_anom]).sample(frac=1)\
                .reset_index(drop=True)
            X = data_sample.iloc[:, :-2]
            y = data_sample.iloc[:, -2]                
            skf = StratifiedKFold(n_splits=3)
            for train_index, test_index in skf.split(X, y):
                # Logistic Regression:
                X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
                y_train, y_test = y[train_index], y[test_index]
                lr = LogisticRegression()
                lr.fit(X_train, y_train)
                y_proba = lr.predict_proba(X_test)
                roc_auc_lr.append(roc_auc_score(y_test, y_proba[:, 1]))
                # Random Forests:
                rf =  RandomForestClassifier()
                rf.fit(X_train, y_train)
                y_proba_rf = rf.predict_proba(X_test)
                roc_auc_rf.append(roc_auc_score(y_test, y_proba_rf[:, 1])) 
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
            
            results_point_difficulty[dataset]['lr'][np.round(i / n_datasets, 2)] = \
                np.mean(roc_auc_lr)
            results_point_difficulty[dataset]['rf'][np.round(i / n_datasets, 2)] = \
                np.mean(roc_auc_rf)
            results_point_difficulty[dataset]['gbm'][np.round(i / n_datasets, 2)] = \
                np.mean(roc_auc_gbm)
            results_point_difficulty[dataset]['iforest'][np.round(i / n_datasets, 2)] = \
                np.mean(roc_auc_iforest)
            results_point_difficulty[dataset]['lof'][np.round(i / n_datasets, 2)] = \
                np.mean(roc_auc_lof)
    timestr = time.strftime("%H%M%S")
    name = 'results_point_difficulty_{}'.format(timestr)
    save_results(results_point_difficulty, name)
    
# Plot ROC AUC:
def plot_results_point_difficulty(data_original, results_point_difficulty):
    """Plot results for all original dataset over spectrum of generated
       datasets with various point difficulties
       Input:  * results_point_difficulty: dict with roc_auc score for each
                 generated dataset
    """
    roc_auc_lr, roc_auc_rf, roc_auc_gbm, roc_auc_iforest, roc_auc_lof = [], [], [], [], []
    for dataset in data_original.keys():
        x = np.array(list(results_point_difficulty[dataset]['lr'].keys()))
        y_lr = np.array(list(results_point_difficulty[dataset]['lr'].values()))
        y_rf = np.array(list(results_point_difficulty[dataset]['rf'].values()))
        y_gbm = np.array(list(results_point_difficulty[dataset]['gbm'].values()))
        y_iforest = np.array(list(results_point_difficulty[dataset]['iforest'].values()))
        y_lof = np.array(list(results_point_difficulty[dataset]['lof'].values()))
        roc_auc_lr.append(y_lr)
        roc_auc_rf.append(y_rf)
        roc_auc_gbm.append(y_gbm)
        roc_auc_iforest.append(y_iforest)
        roc_auc_lof.append(y_lof)
        plt.plot(x, y_lr, color='blue')
        plt.plot(x, y_rf, color='lightblue')
        plt.plot(x, y_gbm, color='green')
        plt.plot(x, y_iforest, color='red')
        plt.plot(x, y_lof, color='orange')
        plt.title('ROC AUC - {}'.format(dataset))
    plt.show()
    
    mean_roc_auc_lr = np.mean(np.array(roc_auc_lr), axis=0)
    plt.plot(x, mean_roc_auc_lr, color='blue')
    mean_roc_auc_rf = np.mean(np.array(roc_auc_rf), axis=0)
    plt.plot(x, mean_roc_auc_rf, color='lightblue')
    mean_roc_auc_gbm = np.mean(np.array(roc_auc_gbm), axis=0)
    plt.plot(x, mean_roc_auc_gbm, color='green')
    mean_roc_auc_iforest = np.mean(np.array(roc_auc_iforest), axis=0)
    plt.plot(x, mean_roc_auc_iforest, color='red')
    mean_roc_auc_lof = np.mean(np.array(roc_auc_lof), axis=0)
    plt.plot(x, mean_roc_auc_lof, color='orange')
    plt.title('ROC AUC - mean over all data sets'.format(dataset))
    plt.show()  


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
        results_semantic_variation[dataset]['lr'] = dict()
        results_semantic_variation[dataset]['rf'] = dict()
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
            roc_auc_lr, roc_auc_rf, roc_auc_gbm, roc_auc_iforest, roc_auc_lof = \
                [], [], [], [], []
            data_sample = pd.concat([data_reg, dataset_anom.iloc[:, :-1]]).\
                sample(frac=1).reset_index(drop=True)
            X = data_sample.iloc[:, :-2]
            y = data_sample.iloc[:, -2]                
            skf = StratifiedKFold(n_splits=3)
            for train_index, test_index in skf.split(X, y):
                # Logistic Regression:
                X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
                y_train, y_test = y[train_index], y[test_index]
                lr = LogisticRegression()
                lr.fit(X_train, y_train)
                y_proba = lr.predict_proba(X_test)
                roc_auc_lr.append(roc_auc_score(y_test, y_proba[:, 1]))
                # Random Forests:
                rf =  RandomForestClassifier()
                rf.fit(X_train, y_train)
                y_proba_rf = rf.predict_proba(X_test)
                roc_auc_rf.append(roc_auc_score(y_test, y_proba_rf[:, 1]))                
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
            
            results_semantic_variation[dataset]['lr'][np.round(i / n_datasets, 2)] = \
                np.mean(roc_auc_lr)
            results_semantic_variation[dataset]['rf'][np.round(i / n_datasets, 2)] = \
                np.mean(roc_auc_rf)
            results_semantic_variation[dataset]['gbm'][np.round(i / n_datasets, 2)] = \
                np.mean(roc_auc_gbm)
            results_semantic_variation[dataset]['iforest'][np.round(i / n_datasets, 2)] = \
                np.mean(roc_auc_iforest)
            results_semantic_variation[dataset]['lof'][np.round(i / n_datasets, 2)] = \
                np.mean(roc_auc_lof)
    timestr = time.strftime("%H%M%S")
    name = 'results_semantic_variation_{}'.format(timestr)
    save_results(results_semantic_variation, name)

# Plot ROC AUC: 
def plot_results_semanitc_variation(data_original, results_semanitc_variation):
    """Plot results for all original dataset over spectrum of generated
       datasets with various semanitc variations
       Input:  * results_semanitc_variation: dict with roc_auc score for each
                 generated dataset
    """                                
    roc_auc_lr, roc_auc_rf, roc_auc_gbm, roc_auc_iforest, roc_auc_lof = [], [], [], [], []
    for dataset in data_original.keys():
        x = np.array(list(results_semanitc_variation[dataset]['lr'].keys()))
        y_lr = np.array(list(results_semanitc_variation[dataset]['lr'].values()))
        y_rf = np.array(list(results_semanitc_variation[dataset]['rf'].values()))
        y_gbm = np.array(list(results_semanitc_variation[dataset]['gbm'].values()))
        y_iforest = np.array(list(results_semanitc_variation[dataset]['iforest'].values()))
        y_lof = np.array(list(results_semanitc_variation[dataset]['lof'].values()))
        roc_auc_lr.append(y_lr)
        roc_auc_rf.append(y_rf)
        roc_auc_gbm.append(y_gbm)
        roc_auc_iforest.append(y_iforest)
        roc_auc_lof.append(y_lof)
        plt.plot(x, y_lr, color='blue')
        plt.plot(x, y_rf, color='lightblue')
        plt.plot(x, y_gbm, color='green')
        plt.plot(x, y_iforest, color='red')
        plt.plot(x, y_lof, color='orange')
        plt.title('ROC AUC - {}'.format(dataset))
    plt.show()
    
    mean_roc_auc_lr = np.mean(np.array(roc_auc_lr), axis=0)
    plt.plot(x, mean_roc_auc_lr, color='blue')
    mean_roc_auc_rf = np.mean(np.array(roc_auc_rf), axis=0)
    plt.plot(x, mean_roc_auc_rf, color='lightblue')
    mean_roc_auc_gbm = np.mean(np.array(roc_auc_gbm), axis=0)
    plt.plot(x, mean_roc_auc_gbm, color='green')
    mean_roc_auc_iforest = np.mean(np.array(roc_auc_iforest), axis=0)
    plt.plot(x, mean_roc_auc_iforest, color='red')
    mean_roc_auc_lof = np.mean(np.array(roc_auc_lof), axis=0)
    plt.plot(x, mean_roc_auc_lof, color='orange')
    plt.title('ROC AUC - mean over all data sets'.format(dataset))
    plt.show() 


def main():
    # Prepare data
    print('preparing data ...')
    data_original = dict()
    
    # Prepare credit card data set:
    df = pd.read_csv('data/creditcard.csv')
    df = df.drop(['Time'], axis=1)
    sc = StandardScaler()
    df.iloc[:, :-1] = sc.fit_transform(df.iloc[:, :-1])
    df['point_difficulty'] = point_difficulty(df)
    
    ind_reg = df[df.iloc[:, -2]==0].index
    ind_anom = df[df.iloc[:, -2]==1].index
    regular_size = 10000   # only use 10,000 regular samples
    data_original['credit'] = dict()
    data_original['credit']['regular'] = df.iloc[ind_reg, :].sample(n=regular_size)
    data_original['credit']['anom'] = df.iloc[ind_anom, :]
    
    # Prepare caravan insurance data set:
    df = pd.read_csv('data/caravan-insurance-challenge.csv')
    df = df.drop(['ORIGIN'], axis=1)
    sc = StandardScaler()
    df.iloc[:, :-1] = sc.fit_transform(df.iloc[:, :-1])
    df['point_difficulty'] = point_difficulty(df)
    
    ind_reg = df[df.iloc[:, -2]==0].index
    ind_anom = df[df.iloc[:, -2]==1].index
    data_original['caravan'] = dict()
    data_original['caravan']['regular'] = df.iloc[ind_reg, :]
    data_original['caravan']['anom'] = df.iloc[ind_anom, :]
    
    # Relative frequency:
    anom_freq = np.zeros(11)
    anom_freq[:2] = [0.001, 0.0025]
    anom_freq[2:] = np.linspace(0.005, 0.045, 9)
    print('training datasets with different relative frequencues ...')
    results_relative_frequency(data_original, anom_freq=anom_freq)

    # Point difficulty:
    anom_freq = 0.01
    n_datasets = 10
    print('training datasets with different point difficulties ...')
    results_point_difficulty(data_original, anom_freq=anom_freq, n_datasets=n_datasets)

    # Semantic variance:
    anom_freq = 0.01
    n_datasets = 10
    print('training datasets with different semantic variances...')
    results_semanitc_variation(data_original, anom_freq=anom_freq, n_datasets=n_datasets)
    
    # Load results:
    pkl_relative_freq = glob.glob('results/results_relative_freq_*')
    assert(len(pkl_relative_freq) == 1)
    results_relative_frequency_ = load_results(pkl_relative_freq[0])
    pkl_point_difficulty = glob.glob('results/results_point_difficulty_*')
    assert(len(pkl_point_difficulty) == 1)
    results_point_difficulty_ = load_results(pkl_point_difficulty[0])
    pkl_semantic_variation = glob.glob('results/results_semantic_variation_*')
    assert(len(pkl_semantic_variation) == 1)
    results_semantic_variation_ = load_results(pkl_semantic_variation[0])
    
    # Plot results:
    plot_results_relative_frequency(data_original, results_relative_frequency_)
    plot_results_point_difficulty(data_original, results_point_difficulty_)
    plot_results_semanitc_variation(data_original, results_semantic_variation_)
    
    
if __name__ == '__main__':
    main()


