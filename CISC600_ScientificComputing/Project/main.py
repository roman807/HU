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
from sklearn.preprocessing import StandardScaler
import pickle
import glob
import time
import os
os.chdir('/home/roman/Documents/HU/CISC600_ScientificComputing/Project/')

import semantic_variation
import relative_frequency
import point_difficulty
import inputs


def load_results(name):
    with open(name, 'rb') as f:
        return pickle.load(f)

def main():
    settings = inputs.create_settings()
    # Prepare data
    print('preparing data ...')
    data_original = dict()
    
    # Prepare credit card data set:
    if settings['datasets']['credit']:
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
    if settings['datasets']['caravan']:
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
    if settings['datasets']['kddcup99']:
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
    
    # Relative frequency:
    #settings_relative_frequency = settings['settings_relative_frequency']
    if settings['settings_relative_frequency']['train']:
        print('training datasets with different relative frequencies ...')
        start = time.time()
        relative_frequency.results_relative_frequency(
                data_original, 
                settings)
        end = time.time()
        t = (end - start) / 60
        print('time train relative frequency: {} minutes'.format(t))
    
    ### Point difficulty:
    if settings['settings_point_difficulty']['train']:
        print('training datasets with different point difficulties ...')
        start = time.time()
        point_difficulty.results_point_difficulty(
                data_original, 
                settings)
        end = time.time()
        t = (end - start) / 60
        print('time train point difficulty: {} minutes'.format(t))
    
    ### Semantic variance:
    if settings['settings_semantic_variance']['train']:
        print('training datasets with different semantic variances ...')
        start = time.time()
        semantic_variation.results_semanitc_variation(
                data_original, 
                settings)
        end = time.time()
        t = (end - start) / 60
        print('time train sematic variance: {} minutes'.format(t))
    
    # Load results:
    if settings['settings_relative_frequency']['plot_results']:
        model_names, results = [], []
        for model in settings['settings_relative_frequency']['models_load'].keys():
            if settings['settings_relative_frequency']['models_load'][model]:
                model_names.append(model)
                pkl_model = glob.glob(settings['results_dir'] + \
                                      '/results_relative_frequency_{}*'.format(model))
                print(pkl_model)
                assert(len(pkl_model) == 1)
                results.append(load_results(pkl_model[0]))
        # Plot results relative frequency:
        relative_frequency.plot_results_relative_frequency(data_original, results, 
                                                           model_names, settings)

    if settings['settings_point_difficulty']['plot_results']:
        model_names, results = [], []
        for model in settings['settings_point_difficulty']['models_load'].keys():
            if settings['settings_point_difficulty']['models_load'][model]:
                model_names.append(model)
                pkl_model = glob.glob(settings['results_dir'] + \
                                      '/results_point_difficulty_{}*'.format(model))
                print(pkl_model)
                assert(len(pkl_model) == 1)
                results.append(load_results(pkl_model[0]))
        # Plot results point difficulty:
        point_difficulty.plot_results_point_difficulty(data_original, results, 
                                                           model_names, settings)
    
    if settings['settings_semantic_variance']['plot_results']:
        model_names, results = [], []
        for model in settings['settings_semantic_variance']['models_load'].keys():
            if settings['settings_semantic_variance']['models_load'][model]:
                model_names.append(model)
                pkl_model = glob.glob(settings['results_dir'] + \
                                      '/results_semantic_variance_{}*'.format(model))
                print(pkl_model)
                assert(len(pkl_model) == 1)
                results.append(load_results(pkl_model[0]))
        # Plot results semantic variance:
        semantic_variation.plot_results_semantic_variance(data_original, results, 
                                                           model_names, settings)
        
if __name__ == '__main__':
    main()

