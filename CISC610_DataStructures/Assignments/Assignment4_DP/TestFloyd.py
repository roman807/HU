#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Roman Moser, 5/30/19

"""
Test Floyd's algorithm
"""

#import os
#os.chdir('/home/roman/Documents/HU/CISC610_DataStructures/Assignments/Assignment4_DP')
import math
INF = math.inf

from Floyd import Floyd


def main():    
    C = [[0, 9, INF, 1, INF],
         [3, 0, INF, 5, 3],
         [INF, 3, 0, INF, 4],
         [2, INF, 3, 0, INF],
         [INF, 2, 2, INF, 0]]
    print(C)
    floyd = Floyd(C)
    D, P = floyd.floyd()
    print('D: ', D)
    print('P: ', P)

if __name__ == '__main__':
    main()
    