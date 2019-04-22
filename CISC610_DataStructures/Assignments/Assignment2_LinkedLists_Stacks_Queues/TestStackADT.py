#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Roman Moser, 4/8/19

"""
Test all methods of class StackADT
"""

#import os
#os.chdir('/home/roman/Documents/HU/CISC 610 - Data Structures/Assignments/Assignment2')

from StackADT import StackADT

def main():
    print('* create stack with capacity of 3')
    stack = StackADT(max_ = 3)
    print('* check if stack is empty (should be True):')
    print(stack.isEmpty())
    print('* check if stack is full (should be False):')
    print(stack.isFull())
    print('* push 3 items on the stack (1, 2, 3) and print stack')
    stack.push(1)
    stack.push(2)
    stack.push(3)
    stack.items.printNextList()
    print('* check if stack is full (should be True)')
    print(stack.isFull())
    print('* check stack size (should be 3)')
    print(stack.size())
    print('* pop all three items from stack (should yield 3, 2, 1)')
    print(stack.pop())
    print(stack.pop())
    print(stack.pop())
    print('* check if stack is empty (should be True)')
    print(stack.isEmpty())

if __name__ == '__main__':
    main()