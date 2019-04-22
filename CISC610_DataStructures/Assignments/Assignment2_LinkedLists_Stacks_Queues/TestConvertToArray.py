#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Roman Moser, 4/2/19

"""
Test ConvertToArray
"""

from DLinkedListADT import DLinkedListADT
from ConvertToArray import ConvertToArray

def main():
    print('create doubly linked list and print it:')
    dl = DLinkedListADT()
    dl.addFirst(6)
    dl.addFirst(5)
    dl.addFirst(4)
    dl.addFirst(3)
    dl.addFirst(2)
    dl.addFirst(1)
    dl.printNextList()
    
    print('convert list (of length 6) to array (of size 2*3):')
    cta = ConvertToArray()
    print(cta.convert_to_array(dl, 2, 3))

if __name__ == '__main__':
    main()