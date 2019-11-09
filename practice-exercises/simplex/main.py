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

def get_entering_var(non_basic_vars, equations):
    min_ = 0
    for i, element in enumerate(equations[0, :]):
        if i in non_basic_vars:
            if element < min_:
                entering_var = i
    return entering_var

def get_leaving_var(
        entering_var, 
        system_of_equations,
        solution,
        basic_vars
    ):
    intercept = np.array(solution[1:]) / system_of_equations[1:, entering_var]
    min_intercept = np.argwhere(intercept == min(intercept[np.where(intercept > 0)]))[0][0]
    return basic_vars[min_intercept]

def get_new_basic_vars(
        basic_vars,
        non_basic_vars,
        entering_var,
        leaving_var
    ):
    basic_vars = [i if i != leaving_var else entering_var
                       for i in basic_vars]
    non_basic_vars = [i if i != entering_var else leaving_var
                           for i in non_basic_vars]
    return basic_vars, non_basic_vars

def substitute_entering_var(
        system_of_equations, solution,
        entering_var, 
        row_ind_leaving_var
    ):
    for i in range(system_of_equations.shape[0]):
        if i != row_ind_leaving_var:
            solution[i] -= solution[row_ind_leaving_var] * \
                system_of_equations[i][entering_var]
            system_of_equations[i] += system_of_equations[i][entering_var] * \
                (- system_of_equations[row_ind_leaving_var, :])
    return system_of_equations, solution

def main():
    # DEFINE PROBLEM:
    utility_function = [150, 175]
    constraints = [
            [7, 11],
            [10, 8],
            [1, 0],
            [0, 1]
    ]
    rhs = [77, 80, 9, 6]
    
    n_structural_vars = len(utility_function)
    n_slack_vars = len(constraints)
    n_total_vars = n_structural_vars + n_slack_vars
    
    non_basic_vars = [i for i in range(n_structural_vars)]
    basic_vars = [i for i in range(n_total_vars) if i not in non_basic_vars]
    
    row_1 = np.concatenate([-np.array(utility_function), np.zeros(n_slack_vars)])
    for i in range(n_slack_vars):
        for j in range(n_slack_vars):
            if i == j:
                constraints[i].append(1)
            else:
                constraints[i].append(0)
    equations = np.concatenate([row_1.reshape(1, -1), np.array(constraints)], axis=0)
    solution = np.array([0] + rhs).astype(float)
    
    while True:
        entering_var = get_entering_var(
                non_basic_vars, 
                equations
        )
        leaving_var = get_leaving_var(
                entering_var, 
                equations, 
                solution, 
                basic_vars
        )
        row_ind_leaving_var = [i for i in range(len(basic_vars)) \
                                      if basic_vars[i] == leaving_var][0] + 1
                                      
        basic_vars, non_basic_vars = get_new_basic_vars(
                basic_vars,
                non_basic_vars,
                entering_var,
                leaving_var
        )
        
        solution[row_ind_leaving_var] /= equations[row_ind_leaving_var, entering_var]
        equations[row_ind_leaving_var, :] /= equations[row_ind_leaving_var, entering_var]
            
        equations, solution = substitute_entering_var(
                equations, 
                solution, 
                entering_var, 
                row_ind_leaving_var
        )    
        
        if min(equations[0, :]) >= 0:
            break
    
    for i in range(len(utility_function)):
        value = np.round(solution[[j for j in basic_vars if j == i][0] + 1], 3)
        print('x{}: {}'.format(i + 1, value))
    print('utility value: {}'. format(np.round(solution[0], 3)))
    

if __name__ == '__main__':
    main()
    
