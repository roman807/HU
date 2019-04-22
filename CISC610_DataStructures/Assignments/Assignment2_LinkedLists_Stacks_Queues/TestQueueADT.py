#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Roman Moser, 4/8/19

"""
Test all methods of class QueueADT
"""

from QueueADT import QueueADT

def main():
    print('* create queue with capacity of 3')
    queue = QueueADT(max_ = 3)
    print('* check if queue is empty (should be True):')
    print(queue.isEmpty())
    print('* check if queue is full (should be False):')
    print(queue.isFull())
    print('* push 3 items on the queue (1, 2, 3) and print queue')
    queue.enQueue(1)
    queue.enQueue(2)
    queue.enQueue(3)
    queue.items.printNextList()
    print('* check if queue is full (should be True)')
    print(queue.isFull())
    print('* check queue size (should be 3)')
    print(queue.size())
    print('* pop all three items from queue (should yield 1, 2, 3)')
    print(queue.deQueue())
    print(queue.deQueue())
    print(queue.deQueue())
    print('* check if queue is empty (should be True)')
    print(queue.isEmpty())

if __name__ == '__main__':
    main()
