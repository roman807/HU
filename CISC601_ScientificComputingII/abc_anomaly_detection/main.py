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

def main():
    # read inputs:
    with open("inputs.json") as f:
        inputs = json.load(f)
    dataset = inputs['dataset']
    models = inputs['models']
    
    # prepare data:
    df_train = pd.read_csv('data/' + dataset + '_train.csv')
    df_test_A = pd.read_csv('data/' + dataset + '_test_A.csv')
    df_test_U = pd.read_csv('data/' + dataset + '_test_U.csv')
    
    X_train = df_train[df_train.columns[df_train.columns.isin(['label'])==False]]
    X_test_A = df_test_A[df_test_A.columns[df_test_A.columns.isin(['label'])==False]]
    X_test_U = df_test_U[df_test_U.columns[df_test_U.columns.isin(['label'])==False]]
    y_train = df_train['label']
    y_test_A = df_test_A['label']
    y_test_U = df_test_U['label']
    
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
    
    with open('results/' + str(dataset) + '_' + '_'.join(models) + '.json', 'w') as fp:
        json.dump(results, fp)

if __name__ == '__main__':
    main()
