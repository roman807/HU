#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Roman Moser, 6/19/19

"""
Prepare datasets and run all models. Save results as pickle files in 'results'
and load results to create plots
run with: python3 main.py 
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import glob
import time
import os
os.chdir('/home/roman/Documents/HU/CISC600_ScientificComputing/Project')

import semantic_variation
import relative_frequency
import point_difficulty


def save_results(obj, name):
    with open('results/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f)

def load_results(name):
    with open(name, 'rb') as f:
        return pickle.load(f)

def main():
    # Prepare data
    print('preparing data ...')
    data_original = dict()
    
    # Prepare credit card data set:
    df = pd.read_csv('data/creditcard.csv')
    df = df.drop(['Time'], axis=1)
    sc = StandardScaler()
    df.iloc[:, :-1] = sc.fit_transform(df.iloc[:, :-1])
    df['point_difficulty'] = point_difficulty.point_difficulty(df)
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
    df['point_difficulty'] = point_difficulty.point_difficulty(df)
    ind_reg = df[df.iloc[:, -2]==0].index
    ind_anom = df[df.iloc[:, -2]==1].index
    data_original['caravan'] = dict()
    data_original['caravan']['regular'] = df.iloc[ind_reg, :]
    data_original['caravan']['anom'] = df.iloc[ind_anom, :]
    
    # Prepare KDD Cup 99 data set:
    df = pd.read_csv('data/kddcup99_csv.csv')
    df = df.drop(['protocol_type', 'service', 'flag'], axis=1)
    df.label = df.label.apply(lambda x: 1 if x=='normal' else 0)
    sc = StandardScaler()
    df.iloc[:, :-1] = sc.fit_transform(df.iloc[:, :-1])
    df['point_difficulty'] = point_difficulty.point_difficulty(df)
    ind_reg = df[df.iloc[:, -2]==0].index
    ind_anom = df[df.iloc[:, -2]==1].index
    regular_size = 10000   # only use 10,000 regular samples
    data_original['kddcup99'] = dict()
    data_original['kddcup99']['regular'] = df.iloc[ind_reg, :].sample(n=regular_size)
    data_original['kddcup99']['anom'] = df.iloc[ind_anom, :]
    
    # save prepared data:
    save_results(data_original, 'prepared_data')
    
    # Relative frequency:
    anom_freq = np.zeros(11)
    anom_freq[:2] = [0.001, 0.0025]
    anom_freq[2:] = np.linspace(0.005, 0.045, 9)
    print('training datasets with different relative frequencies ...')
    start = time.time()
    results = relative_frequency.results_relative_frequency(
            data_original, 
            anom_freq=anom_freq)
    end = time.time()
    timestr = time.strftime("%H%M%S")
    name = 'results_relative_freq_{}'.format(timestr)
    save_results(results, name)
    t = (end - start) / 60
    print('time train relative frequency: {} minutes'.format(t))
    
    # Point difficulty:
    anom_freq = 0.005
    n_datasets = 10
    print('training datasets with different point difficulties ...')
    start = time.time()
    point_difficulty.results(data_original, anom_freq=anom_freq, n_datasets=n_datasets)
    timestr = time.strftime("%H%M%S")
    name = 'results_point_difficulty_{}'.format(timestr)
    save_results(results, name)
    end = time.time()
    t = (end - start) / 60
    print('time train point difficulties: {} minutes'.format(t))
    
    # Semantic variance:
    anom_freq = 0.005
    n_datasets = 10
    print('training datasets with different semantic variances...')
    start = time.time()
    results = semantic_variation.results_semanitc_variation(
            data_original, 
            anom_freq=anom_freq, 
            n_datasets=n_datasets)
    timestr = time.strftime("%H%M%S")
    name = 'results_semantic_variation_{}'.format(timestr)
    save_results(results, name)
    end = time.time()
    t = (end - start) / 60
    print('time train semantic variances: {} minutes'.format(t))
    
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
    relative_frequency.plot_results_relative_frequency(data_original, results_relative_frequency_)
    point_difficulty.plot_results_point_difficulty(data_original, results_point_difficulty_)
    semantic_variation.plot_results_semanitc_variation(data_original, results_semantic_variation_)
    
    
if __name__ == '__main__':
    main()

