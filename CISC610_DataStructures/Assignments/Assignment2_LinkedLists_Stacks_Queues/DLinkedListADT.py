#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Roman Moser, 4/2/19

"""
implementation of a doubly linked list
class DLinkedList contains the following classes and mathods:
 * class: Node
 * method: addFirst
 * method: addLast
 * method: deleteFirst
 * method: deleteLast
 * method: deleteAll
 * method: getNext
 * method: size
 * method: printNextList
 * method: printPrevList
"""

class DLinkedListADT():
    def __init__(self):
        self.front = None
        self.rear = None
        self.length = 0
        self.currentNode = None
        
    class Node():
        def __init__(self, key=None, next_=None, prev=None):
            self.key = key
            self.next_ = next_
            self.prev = prev
    
        def getKey(self):
            return self.key
        
        def getNext(self):
            return self.next_    
        
        def getPrev(self):
            return self.prev  
  
    def addFirst(self, data):
        n = self.Node(data)
        if self.front == None:
            self.front = n
            self.rear = n
            self.length = 1
        else:
            n.next_ = self.front
            self.front.prev = n
            self.front = n
            self.length += 1
            
    def addLast(self, data):
        n = self.Node(data)
        if self.front == None:
            self.front = n
            self.rear = n
            self.length = 1
        else:
            self.rear.next_ = n
            n.prev = self.rear
            self.rear = n
            self.length += 1
            
    def deleteFirst(self):
        if self.length == 0:
            return 'cannot delete first element, list is empty'
        n = self.front
        if self.front == self.rear:
            self.front = None
            self.rear = None
            self.length = 0
        else:
            self.front = self.front.next_#.head
            self.front.prev = None
            self.length -= 1
        return n.key

    def deleteLast(self):
        if self.length == 0:
            return 'cannot delete last element, list is empty'
        n = self.rear
        if self.front == self.rear:
            self.front = None
            self.rear = None    
            self.length = 0
        else:
            self.rear = self.rear.prev
            self.rear.next_ = None
            self.length -= 1
        return n.key
    
    def deleteAll(self):
        while self.length != 0:
            self.deleteLast()
    
    def getNext(self):
        if self.length == 0:
            return 'cannot return next element, list is empty'
        if self.currentNode == None:
            self.currentNode = self.front
        else:
            self.currentNode = self.currentNode.getNext()
        return self.currentNode.key
        
    def size(self):
        return self.length
    
    def printNextList(self):
        n = self.front
        while n != None:
            print(n.key)
            n = n.next_

    def printPrevList(self):
        n = self.rear
        while n != None:
            print(n.key)
            n = n.prev
    
    def insertAfter(self, oldKey, newKey):
        if self.size() == 0:
            self.addFirst(newKey)
        else:
            self.currentNode = self.front
            while self.currentNode.key != oldKey:
                if self.currentNode == self.rear:
                    break
                self.currentNode = self.currentNode.next_
            n = self.Node(newKey)
            old = self.currentNode
            tmp = self.currentNode.next_
            self.currentNode.next_ = n
            self.currentNode = self.currentNode.next_
            self.currentNode.next_ = tmp
            self.currentNode.prev = old