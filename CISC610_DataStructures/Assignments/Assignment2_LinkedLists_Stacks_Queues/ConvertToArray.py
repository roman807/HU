#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Roman Moser, 4/2/19

"""
Method that converts doubly linked list into array of certain dimension
"""

class ConvertToArray():        
    def convert_to_array(self, dl, x, y):
        if dl.size() != x * y:
            return 'error: array dimension does not match list size'
        else:
            array = []
            for i in range(x):
                l_ = []
                for j in range(y):
                    l_.append(dl.getNext())
                array.append(l_)
        return array