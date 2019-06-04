#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Roman Moser, 6/2/19

"""
Preprocess census data
"""

import pandas as pd
import os
os.chdir('/home/roman/Documents/HU/CISC600_ScientificComputing/Project')

df = pd.read_csv('data/census-adult.csv')

continent_dict = {'United-States': 'North-America', 
                  'Canada': 'North-America', 
                  'Mexico': 'North-America',
                  'Trinadad&Tobago': 'South-America',
                  'Greece': 'Europe', 
                  'Vietnam': 'Asia',
                  'Taiwan': 'Asia',
                  'India': 'Asia', 
                  'Philippines': 'Asia', 
                  'South': 'South-America', 
                  'Holand-Netherlands': 'Europe',
                  'Puerto-Rico': 'North-America',
                  'Poland': 'Europe',
                  'Iran': 'Asia', 
                  'England': 'Europe', 
                  'Germany': 'Europe', 
                  'Italy': 'Europe',
                  'Japan': 'Asia', 
                  'Hong': 'Asia', 
                  'Honduras': 'South-America',
                  'Cuba': 'South-America', 
                  'Ireland': 'Europe', 
                  'Cambodia': 'Asia',
                  'Peru': 'South-America', 
                  'Nicaragua': 'South-America', 
                  'Dominican-Republic': 'North-America',
                  'Haiti': 'North-America',
                  'El-Salvador': 'South-America', 
                  'Hungary': 'Europe', 
                  'Columbia': 'South-America',
                  'Guatemala': 'South-America', 
                  'Jamaica': 'North-America', 
                  'Ecuador': 'South-America', 
                  'France': 'Europe', 
                  'Yugoslavia': 'Europe',
                  'Scotland': 'Europe',
                  'Portugal': 'Europe',
                  'Laos': 'Asia', 
                  'Thailand': 'Asia', 
                  'Outlying-US(Guam-USVI-etc)': 'North-America',
                  '?': 'Unknown',
                  'China': 'Asia'}
df['continent'] = df['native.country'].apply(lambda x: continent_dict[x])
continent_dummies = pd.get_dummies(df['continent'])
occupation_dummies = pd.get_dummies(df['occupation'])
workclass_dummies = pd.get_dummies(df['workclass'])
marital_status_dummies = pd.get_dummies(df['marital.status'])
relationship_dummies = pd.get_dummies(df['relationship'])
race_dummies = pd.get_dummies(df['race']).iloc[:, 0]
sex_dummies = pd.get_dummies(df['sex']).iloc[:, 0]
larger_50K = df['income'].apply(lambda x: 1 if x == '>50K' else 0)
df = df.drop(['education', 'occupation', 'workclass', 'marital.status', 
              'relationship', 'race', 'sex', 'native.country', 'continent',
              'income'], axis=1)
df = pd.concat([df, continent_dummies, workclass_dummies, marital_status_dummies,
                relationship_dummies, race_dummies, sex_dummies, larger_50K], axis=1)
df.to_csv('data/census_revised.csv')











