#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Roman Moser, 4/8/19

"""
implementation of queue with doubly linked list
class QueueADT contains the following mathods:
 * method: isEmpty: returns True if queue has no items
 * method: isFull: returns True if max capacity of queue is reached
 * method: enQueue: add item to rear of queue
 * method: deQueue: remove and return item from front of queue
 * method: size: return size of queue
"""

from DLinkedListADT import DLinkedListADT

class QueueADT():
    def __init__(self, max_):
        self.items = DLinkedListADT()
        if max_ < 0:
            return 'max_ must be greater than 0'
        self.capacity = max_
    
    def isFull(self):
        if self.items.size() == self.capacity:
            return True
        else:
            return False
        
    def isEmpty(self):
        if self.items.size() == 0:
            return True
        else:
            return False
      
    def enQueue(self, item):
        if self.isFull():
            return 'queue is already full'
        else:
            self.items.addLast(item)
            
    def deQueue(self):
        if self.isEmpty():
            return 'queue is already empty'
        else:
            return self.items.deleteFirst()
        
    def size(self):
        return self.items.size()