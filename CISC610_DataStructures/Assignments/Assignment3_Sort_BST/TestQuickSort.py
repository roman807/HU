#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Roman Moser, 4/22/19

"""
Test QuickSort
"""

#import os
#os.chdir('/home/roman/Documents/HU/CISC610_DataStructures/Assignments/Assignment3_Sort_BST')
import random
import time
import numpy as np
from QuickSort import QuickSort

def main():
    # Sort the following array: [43, 1, 32, 7, 12, 56, 2, 14]
    print('1. Test QuickSort algorithm')
    a = [43, 1, 32, 7, 12, 56, 2, 14]
    print('array before quicksort: {}'.format(a))
    s = QuickSort()
    s.sort(list(a))
    print('array after quicksort: {}\n'.format(s.numbers))
    
    # Test runtimes
    print('2. Test runtimes')
    random.seed(0)
    array_size = 1000
    dataset1 = [random.randint(0, array_size) for i in range(array_size)]
    dataset2 = sorted(dataset1)
    dataset3 = list(reversed(dataset2))
    datasets = [dataset1, dataset2, dataset3]
    
    for pivot in ['middle', 'first', 'last', 'random']:
        print('Test quicksort with {} element as pivot'.format(pivot))
        for i, dataset in enumerate(datasets):
            if i == 0:
                print('time if dataset unsorted:')
            if i == 1:
                print('time if dataset already sorted:')
            if i == 2:
                print('time if dataset sorted in reverse order:')
            times = []
            for i in range(10):
                start = time.time()
                s = QuickSort(pivot=pivot)
                s.sort(list(dataset))
                end = time.time()
                times.append(end - start)
            print(np.mean(times))
        print('---------------')

if __name__ == '__main__':
    main()
