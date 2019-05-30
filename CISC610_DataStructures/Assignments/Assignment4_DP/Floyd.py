#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Roman Moser, 5/30/19

"""
Floyd's algorithm

Given array with weights for paths between nodes, find the shortest path
(min weight) that connects each pair of nodes. INF means that there is no
path between two nodes
"""

import math
import numpy as np

INF = math.inf

class Floyd():
    """Class with method of floyd algorithm
       Arguments:
           C: Matrix of weights for direct paths between nosed
           V: number of nodes
       Outputs:
           D: Array of shortest distances between noddes
           P: Array of intermediate nodes if path not direct
    """
    def __init__(self, C):
        self.C = np.array(C)
        self.V = self.C.shape[0]
        self.D = self.C
        self.P = np.zeros(self.D.shape)

    def floyd(self):
        for k in range(self.V):
            for i in range(self.V):
                for j in range(self.V):
                    if self.D[i, k] + self.D[k, j] < self.D[i, j]:
                        self.D[i, j] = self.D[i, k] + self.D[k, j]
                        self.P[i, j] = k + 1
        return self.D, self.P
    