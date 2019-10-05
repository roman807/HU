#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Roman Moser, 10/5/19

"""
Prepare mnist data set according to Yuki Yamanaka, Tomoharu Iwata,
Hiroshi Takahashi, Masanori Yamada, Sekitoshi Kanai 
https://arxiv.org/pdf/1903.10709.pdf
"""

import pandas as pd
from sklearn.model_selection import train_test_split

import os
os.chdir('/home/roman/Documents/HU/CISC601_ScientificComputingII/abc_anomaly_detection/data')

mnist = pd.read_csv('train.csv')
mnist_train, mnist_test = train_test_split(mnist, test_size=0.5, random_state=123)

# regular, known anomalies (A), unknown anomalies (U):
REGULAR = [1, 3, 5, 7, 9]
ANOM_A = [0, 2, 6, 8]
ANOM_U = [4]

# label 4: unknown anomalies --> exclude from training set:
mnist_train = mnist_test[mnist_test['label'].isin(ANOM_U) == False]

# create test set with only known and only unknown anomalies:
mnist_test_A = mnist_test[mnist_test['label'].isin(ANOM_U) == False]
mnist_test_U = mnist_test[mnist_test['label'].isin(ANOM_A) == False]

# set labels to 0 (regular) and 1 (anomaly):
d = {}
for i in REGULAR:
    d[i] = 0
for i in ANOM_A + ANOM_U:
    d[i] = 1

mnist_train.loc[:, 'label'] = mnist_train['label'].apply(lambda x: d[x])
mnist_test_A.loc[:, 'label'] = mnist_test_A['label'].apply(lambda x: d[x])
mnist_test_U.loc[:, 'label'] = mnist_test_U['label'].apply(lambda x: d[x])

mnist_train.to_csv('mnist_train.csv', index=False)
mnist_test_A.to_csv('mnist_test_A.csv', index=False)
mnist_test_U.to_csv('mnist_test_U.csv', index=False)
