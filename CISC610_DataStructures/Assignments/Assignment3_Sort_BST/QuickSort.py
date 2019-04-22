#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Roman Moser, 4/22/19

"""
implementation of quick sort
 * method sort: calls quicksort if list not empty
 * method quicksort: actual quicksort
 * method exchange: switch values
"""

import random

class QuickSort():
    def __init__(self, pivot='middle'):
        self.numbers = None
        self.pivot = pivot
    
    def sort(self, values):
        if len(values) == 0:
            return
        self.numbers = values
        length = len(values)
        self.quicksort(0, length - 1)
    
    def quicksort(self, low, high):
        i = low
        j = high
        if self.pivot == 'middle':
            pivot = self.numbers[low + int((high - low) / 2)]
        if self.pivot == 'first':
            pivot = self.numbers[low]
        if self.pivot == 'last':
            pivot = self.numbers[high]
        if self.pivot == 'random':
            pivot = self.numbers[random.randint(low, high)]
        while i <= j:
            while self.numbers[i] < pivot:
                i += 1
            while self.numbers[j] > pivot:
                j -= 1
            if i <= j:
                self.exchange(i, j)
                i += 1
                j -= 1
        if low < j:
            self.quicksort(low, j)
        if i < high:
            self.quicksort(i, high)
            
    def exchange(self, i, j):
        tmp = self.numbers[i]
        self.numbers[i] = self.numbers[j]
        self.numbers[j] = tmp

