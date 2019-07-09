#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Roman Moser, 7/3/19

"""
Specify all input settings
"""

import numpy as np
from pathlib import Path
import os

def create_settings():
    # results folder:
    folder_name = 'relative_frequency'
    
    # datasets:
    datasets = {
            'credit': True,
            'caravan': False,
            'kddcup99': False
            }
    
    # relative frequency settings
    train = False
    plot_results = False
#    anom_freq = np.zeros(11)
#    anom_freq[:2] = [0.001, 0.0025]
#    anom_freq[2:] = np.linspace(0.005, 0.045, 9)
    anom_freq = np.array([0.001, 0.0025, 0.005, 0.01, 0.02, 0.03, 0.04])
    n_random_samples = 2
    models_train = {
            'lr': True,
            'gbm': False,
            'iforest': True,
            'lof': False,
            'autoencoder_unsupervised': False,
            'autoencoder_supervised': False
            }
    models_load = {
            'lr': True,
            'gbm': False,
            'iforest': True,
            'lof': False,
            'autoencoder_unsupervised': False,
            'autoencoder_supervised': False
            }
    
    settings_relative_frequency = {
            'train': train,
            'plot_results': plot_results,
            'anom_freq': anom_freq,
            'n_random_samples': n_random_samples,
            'models_train': models_train,
            'models_load': models_load
        }
    
    ### point difficulty settings:
    train = False
    plot_results = True
    anom_freq = 0.005
    n_datasets = 10
    models_train = {
            'lr': True,
            'gbm': False,
            'iforest': False,
            'lof': True,
            'autoencoder_unsupervised': False,
            'autoencoder_supervised': False
            }
    models_load = {
            'lr': True,
            'gbm': False,
            'iforest': False,
            'lof': True,
            'autoencoder_unsupervised': False,
            'autoencoder_supervised': False
            }

    settings_point_difficulty = {
            'train': train, 
            'plot_results': plot_results,
            'anom_freq': anom_freq,
            'n_datasets': n_datasets,
            'models_train': models_train,
            'models_load': models_load
            }
    
    ### semantic variance settings:
    train = False
    plot_results = True
    anom_freq = 0.005
    n_datasets = 10
    models_train = {
            'lr': True,
            'gbm': False,
            'iforest': False,
            'lof': True,
            'autoencoder_unsupervised': False,
            'autoencoder_supervised': False
            }
    models_load = {
            'lr': True,
            'gbm': False,
            'iforest': False,
            'lof': True,
            'autoencoder_unsupervised': False,
            'autoencoder_supervised': False
            }
        
    settings_semantic_variance = {
            'train': train, 
            'plot_results': plot_results,
            'anom_freq': anom_freq,
            'n_datasets': n_datasets,
            'models_train': models_train,
            'models_load': models_load
            }
    
    results_dir = 'results/'
    plots_dir = os.path.join(results_dir, 'plots')
    settings = {
            'folder_name': folder_name,
            'results_dir': results_dir,
            'plots_dir': plots_dir,
            'datasets': datasets,
            'settings_relative_frequency': settings_relative_frequency,
            'settings_point_difficulty': settings_point_difficulty,
            'settings_semantic_variance': settings_semantic_variance
            }
    return settings

        