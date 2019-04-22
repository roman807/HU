#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Roman Moser, 4/8/19

"""
implementation of stack with doubly linked list
class StackADT contains the following mathods:
 * method: isEmpty: returns True if stack has no items
 * method: isFull: returns True if max capacity of stack is reached
 * method: push: add item to rear of stack
 * method: pop: remove and return item from rear of stack
 * method: size: return size of stack
"""

from DLinkedListADT import DLinkedListADT

class StackADT():
    def __init__(self, max_):
        self.items = DLinkedListADT()
        if max_ < 0:
            return 'max_ must be greater than 0'
        self.capacity = max_
        
    def isEmpty(self):
        if self.items.size() == 0:
            return True
        else:
            return False
    
    def isFull(self):
        if self.items.size() == self.capacity:
            return True
        else:
            return False        
    
    def push(self, item):
        if self.isFull():
            return 'stack is already full'
        else:
            self.items.addLast(item)
            
    def pop(self):
        if self.isEmpty():
            return 'stack is already empty'
        else:
            return self.items.deleteLast()
        
    def size(self):
        return self.items.size()
    