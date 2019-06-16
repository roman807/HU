#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Roman Moser, 6/16/19

"""
CompareSorts
compare ExchangeSort and InsertionSort in terms of:
    * runtime
    * comparison of keys
    * assignment of records
"""

import numpy as np
import random
import time
#import os
#os.chdir('/home/roman/Documents/HU/CISC610_DataStructures/Assignments/Assignment5')

from ExchangeSort import ExchangeSort
from InsertionSort import InsertionSort

def evaluate(sort, sorted_array, prev, algorithm, time_sort, n):
    print('Evaluate {} when sorting {} array with {} elements:'.format(algorithm, prev, n))
    assert sort.s == sorted_array, 'Error: sorted array not correct'
    print('time to sort: {} seconds'.format(np.round(time_sort, 3)))
    print('number of keys compared: {}'.format(sort.comp_keys))
    print('total number of assignment of records: {}\n'.format(sort.assignment_of_records))

def main():
    n = 3000   # number of elements of tested arrays
    sorted_array = list(np.arange(n))

    ### Already sorted arrays:
    # ExchangeSort
    original_array = list(np.arange(n))
    exchange_sort = ExchangeSort(original_array)
    start = time.time()
    exchange_sort.sort()
    end = time.time()
    time_exchange_sort = end - start
    evaluate(exchange_sort, sorted_array, 'already sorted', 'ExchangeSort', time_exchange_sort, n)
    # InsertionSort
    original_array = list(np.arange(n))
    insertion_sort = InsertionSort(original_array)
    start = time.time()
    insertion_sort.sort()
    end = time.time()
    time_insertion_sort = end - start
    evaluate(insertion_sort, sorted_array, 'already sorted', 'InsertionSort', time_insertion_sort, n)
    
    ### Sorted in opposite order:
    # ExchangeSort
    original_array = list(np.arange(n))[::-1]
    exchange_sort = ExchangeSort(original_array)
    start = time.time()
    exchange_sort.sort()
    end = time.time()
    time_exchange_sort = end - start
    evaluate(exchange_sort, sorted_array, 'opposite sorted', 'ExchangeSort', time_exchange_sort, n)
    # InsertionSort
    original_array = list(np.arange(n))[::-1]
    insertion_sort = InsertionSort(original_array)
    start = time.time()
    insertion_sort.sort()
    end = time.time()
    time_insertion_sort = end - start
    evaluate(insertion_sort, sorted_array, 'opposite sorted', 'InsertionSort', time_insertion_sort, n)    
    
    ### Unsorted arrays:
    # ExchangeSort
    original_array = random.sample(range(n), n)
    exchange_sort = ExchangeSort(original_array)
    start = time.time()
    exchange_sort.sort()
    end = time.time()
    time_exchange_sort = end - start
    evaluate(exchange_sort, sorted_array, 'unsorted', 'ExchangeSort', time_exchange_sort, n)
    # InsertionSort
    original_array = random.sample(range(n), n)
    insertion_sort = InsertionSort(original_array)
    start = time.time()
    insertion_sort.sort()
    end = time.time()
    time_insertion_sort = end - start
    evaluate(insertion_sort, sorted_array, 'unsorted', 'InsertionSort', time_insertion_sort, n)


if __name__ == '__main__':
    main()