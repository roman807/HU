#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Roman Moser, 10/5/19

"""
Train and predict known and unknown anomalies
"""

import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from time import time

import os
os.chdir('/home/roman/Documents/HU/CISC601_ScientificComputingII/abc_anomaly_detection')

def main():
    # read inputs:
    with open("inputs.json") as f:
        inputs = json.load(f)
    dataset = inputs['dataset']
    models = inputs['models']
    
    # prepare data:
    data_train = pd.read_csv('data/' + dataset + '_train.csv')
    data_test_A = pd.read_csv('data/' + dataset + '_test_A.csv')
    data_test_U = pd.read_csv('data/' + dataset + '_test_U.csv')
    X_train = data_train.iloc[:, 1:]
    X_test_A = data_test_A.iloc[:, 1:]
    X_test_U = data_test_U.iloc[:, 1:]
    y_train = data_train.iloc[:, 0]
    y_test_A = data_test_A.iloc[:, 0]
    y_test_U = data_test_U.iloc[:, 0]
    
    # train & predict:
    results = {}
    for model in models:
        results[model] = {}
        print('train ' + str(model))
        i = __import__('models.' + model, fromlist=[''])
        clf = getattr(i, model)(X_train, y_train)
        start = time()
        clf.fit()
        end = time()
        results[model]['time'] = end - start
        y_train_pred = clf.predict(X_train)
        y_test_A_pred = clf.predict(X_test_A)
        y_test_U_pred = clf.predict(X_test_U)
        results[model]['train_roc_auc'] = roc_auc_score(y_train, y_train_pred)
        results[model]['test_A_roc_auc'] = roc_auc_score(y_test_A, y_test_A_pred)
        results[model]['test_U_roc_auc'] = roc_auc_score(y_test_U, y_test_U_pred)
    
    # print & save results:
    for model in models:
        print('********** Results ' + str(model) + ' **********')
        print('training time:', np.round(results[model]['time']))
        print('AUC ROC training data:         ', results[model]['train_roc_auc'])
        print('AUC ROC test known anomalies:  ', results[model]['test_A_roc_auc'])
        print('AUC ROC test unknown anomalies:', results[model]['test_U_roc_auc'], '\n')
    
    with open('results/results_' + '_'.join(models) + '.json', 'w') as fp:
        json.dump(results, fp)

if __name__ == '__main__':
    main()
