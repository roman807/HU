#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Roman Moser, 4/25/19

"""
Test BST2 (Binary Search Tree)
BST2 code includes:
    * status for node (True/False)
    * deleteFalse operation that deletes all nodes of BST2 with status=False
"""

#import os
#os.chdir('/home/roman/Documents/HU/CISC610_DataStructures/Assignments/Assignment3_Sort_BST')
from BST2 import BST2

def main():
    print('create BST with elements: 10, 5, 2, 6, 1, 4, 20, 15')
    print('(Nodes with key 6, 4 and 20 have status=False)')
    bst = BST2(10, True)
    bst.insert(5, True)
    bst.insert(2, True)
    bst.insert(6, False)
    bst.insert(1, True)
    bst.insert(4, False)
    bst.insert(20, False)
    bst.insert(15, True)
    print('traverse tree in order and print all keys (expected output: 1, 2, 4, 5, 6, 10, 15, 20):')
    bst.traverseInOrder()
    print('delete nodes with status=False and traverse tree (expected output: 1, 2, 5, 10, 15)')
    bst.deleteFalse()
    bst.traverseInOrder()

if __name__ == '__main__':
    main()