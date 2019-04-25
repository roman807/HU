#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Roman Moser, 4/25/19

"""
Binary search tree (BST)
BST2 code includes:
    * status for node (True/False)
    * deleteFalse operation that deletes all nodes of BST2 with status=False
"""

class Node2():
    def __init__(self, key, status):
        self.key = key
        self.status = status
        self.left = None
        self.right = None
    
class BST2():
    def __init__(self, root, status=False):
        self.__root = Node2(root, status)
        self.to_delete = []
    
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
    
    def insert(self, data, status=False):
        self.__insertR(self.__root, data, status)

    def __insertR(self, root, key, status):
        if root == None:
            return Node2(key, status)
        if key < root.key:
            if root.left == None:
                root.left = Node2(key, status)
            else:
                self.__insertR(root.left, key, status)
        elif key > root.key:
            if root.right == None:
                root.right = Node2(key, status)
            else:
                self.__insertR(root.right, key, status)

    def delete(self, data):
        self.__deleteR(self.__root, data)
    
    def __deleteR(self, root, key):
        if root == None:
            return root
        if key < root.key:
            root.left = self.__deleteR(root.left, key)
        elif key > root.key:
            root.right = self.__deleteR(root.right, key)
        else:
            if root.left == None:
                return root.right
            elif root.right == None:
                return root.left
            else:
                root.key = self.__minValue(root.right)
                root.right = self.__deleteR(root.right, root.key)
        return root
    
    def deleteFalse(self):
        self.__deleteFalseR(self.__root)
        
    def __deleteFalseR(self, root):
        if root != None:
            self.__deleteFalseR(root.left)
            if root.status == False:
                self.delete(root.key)
            self.__deleteFalseR(root.right)

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
