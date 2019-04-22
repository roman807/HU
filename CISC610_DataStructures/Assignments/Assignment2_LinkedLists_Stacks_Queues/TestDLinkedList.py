#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Roman Moser, 4/2/19

"""
Test all methods of class DLinkedListADT
"""

from DLinkedListADT import DLinkedListADT

def main():
    print('-------------')
    print('Test "addFirst", "addLast", "printNextList" and "printPrevList"')
    print('Add 3, 2, 1 to beginning of list and 4 to end of list, print list:')
    linked_list = DLinkedListADT()
    linked_list.addFirst(3)
    linked_list.addFirst(2)
    linked_list.addFirst(1)
    linked_list.addLast(4)
    
    print('Print list from beginning to end (should be 1, 2, 3, 4)')
    linked_list.printNextList()
    
    print('Print list from end to beginning (should be 4, 3, 2, 1)')
    linked_list.printPrevList()
    
    print('-------------')
    print('Test "size"')
    print('Size of list (should be 4):')
    print(linked_list.size())
    
    print('-------------')
    print('Test "deleteFirst" and "deleteLast"')
    print('delete first and last element of list:')
    linked_list.deleteFirst()
    linked_list.deleteLast()
    
    print('print list from beginning to end (should be: 2, 3)')
    linked_list.printNextList()
    
    print('-------------')
    print('Test "getNext"')
    print('print remaining elements from first to last (should be 2, 3)')
    print(linked_list.getNext())
    print(linked_list.getNext())
    
    print('-------------')
    print('Test "deleteAll"')
    print('delete all elements and print size of list (should be 0)')
    linked_list.deleteAll()
    print(linked_list.size())

    print('-------------')
    print('Test "insertAfter"')
    print('create and print doubly linke list: 1, 2, 3')
    linked_list.addFirst(3)
    linked_list.addFirst(2)
    linked_list.addFirst(1)
    linked_list.printNextList()
    print('insert 5 after 2 and print list again (should be 1, 2, 5, 3)')
    linked_list.insertAfter(2, 5)
    linked_list.printNextList()
    print('insert 9 after 4 and print list again')
    print('(since 4 is not in list, 9 should be added to end of list i.e.: 1, 2, 5, 3, 9)')
    linked_list.insertAfter(4, 9)
    linked_list.printNextList()
    print('delete all items in list, insert 1 and print list again:')
    print('the only element in the list should now be 1')
    linked_list.deleteAll()
    linked_list.insertAfter(4, 1)
    linked_list.printNextList()
    
if __name__ == '__main__':
    main()
    