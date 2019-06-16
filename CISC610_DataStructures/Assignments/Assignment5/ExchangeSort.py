#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Roman Moser, 6/16/19

"""
Implementation of Exchange Sort
 sorting algorithm iterates through list and exchanges elements whenever
 they are not in order
 * Input: array (list of elements)
 * Methods:
     exchange: exchanges two elements
     sort: sorts array with exchange sort algorithm
"""

class ExchangeSort():
    def __init__(self, array):
        self.s = array
        self.n = len(array)
        self.comp_keys = 0
        self.assignment_of_records = 0
    def exchange(self, i, j):
        self.assignment_of_records += 3
        tmp = self.s[i]
        self.s[i] = self.s[j]
        self.s[j] = tmp
    def sort(self):
        for i in range(self.n):
            for j in range(i + 1, self.n):
                self.comp_keys += 1
                if self.s[j] < self.s[i]:
                    self.exchange(i, j)
