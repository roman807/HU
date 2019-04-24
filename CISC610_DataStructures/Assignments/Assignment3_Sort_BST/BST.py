#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Roman Moser, 4/23/19

"""
Binary search tree (BST)
"""

class Node():
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None
    
class BST():
    def __init__(self, root):
        self.__root = Node(root)
    
    def find(self, key):
        return self.__findR(self.__root, key)
    
    def __findR(self, root, key):
        if root == None or key == root.key:
            return root
        if key > root.key:
            return self.__findR(root.right, key)
        else:
            return self.__findR(root.left, key)
    
    def findMin(self):
        return self.__findMinR(self.__root)
    
    def __findMinR(self, root):
        if root.left == None:
            return root
        else:
            return self.__findMinR(root.left)
            
    def findMax(self):
        return self.__findMaxR(self.__root)
    
    def __findMaxR(self, root):
        if root.right == None:
            return root
        else:
            return self.__findMaxR(root.right)        
    
    def insert(self, data):
        self.__insertR(self.__root, data)

    def __insertR(self, root, key):
        if root == None:
            return Node(key)
        if key < root.key:
            if root.left == None:
                root.left = Node(key)
            else:
                self.__insertR(root.left, key)
        elif key > root.key:
            if root.right == None:
                root.right = Node(key)
            else:
                self.__insertR(root.right, key)

    def delete(self, data):
        self.__root = self.__deleteR(self.__root, data)
    
    def __deleteR(self, root, key):
        if root == None:
            return root
        if key < root.key:
            return self.__deleteR(root.left, key)
        elif key > root.key:
            return self.__deleteR(root.right, key)
        else:
            if root.left == None:
                return root.right
            elif root.right == None:
                return root.left
            else:
                root.key = self.__minValue(root.right)
                root.right = self.__deleteR(root.right, root.key)
        return root
    
    def __minValue(self, root):
        minv = root.key
        while root.left != None:
            minv = root.left.key
            root = root.left
        return minv
    
    def traverseInOrder(self):
        return self.__traverseInOrderR(self.__root)
    
    def __traverseInOrderR(self, root):
        if root != None:
            self.__traverseInOrderR(root.left)
            self.__visit(root)
            self.__traverseInOrderR(root.right)
    
    def __visit(self, n):
        print(n.key)
        
    def getRoot(self):
        return self.__root
    

b = BST(10)
b.insert(5)
b.insert(20)
b.insert(1)
b.insert(4)
b._BST__root.key
b._BST__root.left.key
b._BST__root.right.key
b.traverseInOrder()
