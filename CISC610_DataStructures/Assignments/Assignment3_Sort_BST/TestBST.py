#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Roman Moser, 5/30/19

"""
Test BST (Binary Search Tree)
"""

#import os
#os.chdir('/home/roman/Documents/HU/CISC610_DataStructures/Assignments/Assignment3_Sort_BST')
from BST import BST

def main():
    #create BST:
    print('create BST with elements: 10, 5, 2, 6, 1, 4, 20, 15, 25')
    bst = BST(10)
    insert_in_BST = [5, 2, 6, 1, 4, 20, 15, 25]
    for item in insert_in_BST:
        bst.insert(item)
    
    # test find() -> find root with key = 2 and print key:
    print('find root with key=2 and print key:')
    print(bst.find(2).key)
    
    # find min/max:
    print('find node with min key and print key (should be 1):')
    print(bst.findMin().key)
    print('find node with max key and print key (should be 25):')
    print(bst.findMax().key)
    
    # traverse tree / delete nodes:
    print('traverse tree in order and print all keys:')
    bst.traverseInOrder()
    print('delete node with key=5 and traverse tree again:')
    bst.delete(5)
    bst.traverseInOrder()
    
    # get root and print it's key:
    print('get root key (should be 10)')
    print(bst.getRoot().key)

if __name__ == '__main__':
    main()