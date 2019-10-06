#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Roman Moser, 10/6/19

"""
Prepare kdd cup 99 data set according to Yuki Yamanaka, Tomoharu Iwata,
Hiroshi Takahashi, Masanori Yamada, Sekitoshi Kanai 
https://arxiv.org/pdf/1903.10709.pdf
"""

import pandas as pd
import numpy as np
import re
import csv

import os
os.chdir('/home/roman/Documents/HU/CISC601_ScientificComputingII/abc_anomaly_detection/data')

# get features (column names)
with open('kddcup_features') as f:
    feat = f.read()
cols = re.sub(':.*\n', ' ', feat).split(' ')[:-1]
cols.append('label')

# convert text files to csv:
with open('kddcup.data_10_percent') as f:
    lines = (line.replace('.\n','').split(",") for line in f)
    with open('kdd_train.csv', 'w') as out_file:
        writer = csv.writer(out_file)
        writer.writerow(cols)
        writer.writerows(lines)

with open('corrected') as f:
    lines = (line.replace('.\n','').split(",") for line in f)
    with open('kdd_test.csv', 'w') as out_file:
        writer = csv.writer(out_file)
        writer.writerow(cols)
        writer.writerows(lines)

# drop certain columns and duplicate rows:
drop_columns = ['protocol_type', 'service', 'flag']

kdd_train = pd.read_csv('kdd_train.csv')
kdd_train.drop(drop_columns, axis=1, inplace=True)
kdd_train.drop_duplicates(inplace=True)

kdd_test = pd.read_csv('kdd_test.csv')
kdd_test.drop(drop_columns, axis=1, inplace=True)
kdd_test.drop_duplicates(inplace=True)

# known anomalies (anom_A): 'neptune'
# unknown anomalies (anom_U): (R2L attacks) ftp_write, guess_passwd, imap, 
# multihop, named, phf, sendmail, snmpgetattack, snmpguess, warezmaster, worm, 
# xlock, xsnoop, httptunnel

# remove unknown anomalies from training set:
labels = np.unique(np.concatenate([kdd_test['label'].unique(), kdd_train['label'].unique()]))
ANOM_A = ['neptune']
ANOM_U = ['ftp_write', 'guess_passwd', 'imap', 'multihop', 'named', 'phf', 
          'sendmail', 'snmpgetattack', 'snmpguess', 'warezmaster', 'worm',
          'xlock', 'xsnoop', 'httptunnel']
regular = [label for label in labels if label not in ANOM_A + ANOM_U]

# create train set with only known and only unknown anomalies:
kdd_train = kdd_train[kdd_train['label'].isin(ANOM_U) == False]

# create test set with only known and only unknown anomalies:
kdd_test_A = kdd_test[kdd_test['label'].isin(ANOM_U) == False]
kdd_test_U = kdd_test[kdd_test['label'].isin(ANOM_A) == False]

# set labels to 0 (regular) and 1 (anomaly):
d = {}
for i in regular:
    d[i] = 0
for i in ANOM_A + ANOM_U:
    d[i] = 1

kdd_train.loc[:, 'label'] = kdd_train['label'].apply(lambda x: d[x])
kdd_test_A.loc[:, 'label'] = kdd_test_A['label'].apply(lambda x: d[x])
kdd_test_U.loc[:, 'label'] = kdd_test_U['label'].apply(lambda x: d[x])

kdd_train.to_csv('kdd_train.csv', index=False)
kdd_test_A.to_csv('kdd_test_A.csv', index=False)
kdd_test_U.to_csv('kdd_test_U.csv', index=False)
