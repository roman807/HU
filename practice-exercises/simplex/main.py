#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Roman Moser, 11/9/19

"""
Implementation of simplex optimization algorithm according to
"Numerical Methods for Engineers" by Steven C. Chapra, Raymond P. Canale,
Seventh Edition, starting page 396
"""

import numpy as np
from simplex_algorithm import Simplex

# Define inputs:
utility_function = [150, 175]
constraints_lhs = [[7, 11], [10, 8], [1, 0], [0, 1]]
constraints_rhs = [77, 80, 9, 6]

def main():   
    simplex = Simplex(utility_function, constraints_lhs, constraints_rhs)
    simplex.run()
    
    print('solution found in {} iterations'.format(simplex.n_iterations))
    for i in range(len(utility_function)):
        value = np.round(simplex.solution[[j for j in simplex.basic_vars 
                                           if j == i][0] + 1], 3)
        print('x{}: {}'.format(i + 1, value))
    print('utility value: {}'. format(np.round(simplex.solution[0], 3)))


if __name__ == '__main__':
    main()
