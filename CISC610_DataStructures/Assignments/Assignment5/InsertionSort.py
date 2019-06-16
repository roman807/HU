#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Roman Moser, 6/16/19

"""
Implementation of Insertion Sort
 sorting algorithm iterates through list and inserts each new element at
 correct location in sorted part of list
 * Input: array (list of elements)
 * Methods:
     sort: sorts array with insertion sort algorithm
"""

class InsertionSort():
    def __init__(self, array):
        self.s = array
        self.n = len(array)
        self.comp_keys = 0
        self.assignment_of_records = 0
    def sort(self):
        for i in range(1, self.n):
            x = self.s[i]
            j = i - 1
            self.comp_keys += 1
            while j >= 0 and self.s[j] > x:
                self.assignment_of_records += 1
                self.s[j + 1] = self.s[j]
                j -= 1
                self.comp_keys += 1
            self.assignment_of_records += 1
            self.s[j + 1] = x
