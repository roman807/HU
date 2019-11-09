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

class Simplex:
    def __init__(self, utility_function, constraints_lhs, constraints_rhs):
        n_structural_vars = len(utility_function)
        n_slack_vars = len(constraints_lhs)
        n_total_vars = n_structural_vars + n_slack_vars
        self.n_iterations = 0
        self.non_basic_vars = [i for i in range(n_structural_vars)]
        self.basic_vars = [i for i in range(n_total_vars) \
                           if i not in self.non_basic_vars]
        self.get_initial_table(
                utility_function,
                n_slack_vars,
                constraints_lhs,
                constraints_rhs
        )

    def get_initial_table(
            self,
            utility_function,
            n_slack_vars,
            constraints_lhs,
            constraints_rhs
    ):
            row_0 = np.concatenate([-np.array(utility_function), np.zeros(n_slack_vars)])
            for i in range(n_slack_vars):
                for j in range(n_slack_vars):
                    if i == j:
                        constraints_lhs[i].append(1)
                    else:
                        constraints_lhs[i].append(0)
            self.equations = np.concatenate([row_0.reshape(1, -1),
                                             np.array(constraints_lhs)], axis=0)
            self.solution = np.array([0] + constraints_rhs).astype(float)

    def get_entering_var(self):
        min_ = 0
        for i, element in enumerate(self.equations[0, :]):
            if i in self.non_basic_vars:
                if element < min_:
                    entering_var = i
        return entering_var

    def get_leaving_var(self, entering_var):
        with np.errstate(divide='ignore'):
            intercept = np.array(self.solution[1:]) / self.equations[1:, entering_var]
        min_intercept = np.argwhere(intercept == min(intercept[np.where(intercept > 0)]))[0][0]
        return self.basic_vars[min_intercept]

    def update_basic_and_non_basic_vars(self, entering_var, leaving_var):
        self.basic_vars = [i if i != leaving_var else entering_var
                           for i in self.basic_vars]
        self.non_basic_vars = [i if i != entering_var else leaving_var
                               for i in self.non_basic_vars]

    def update_equations(self, entering_var, leaving_var, row_ind_leaving_var):
        # update pivot element:
        self.solution[row_ind_leaving_var] /= self.equations[row_ind_leaving_var, entering_var]
        self.equations[row_ind_leaving_var, :] /= self.equations[row_ind_leaving_var, entering_var]
        # eliminate coefficients of entering var in other equations:
        for i in range(self.equations.shape[0]):
            if i != row_ind_leaving_var:
                self.solution[i] -= self.solution[row_ind_leaving_var] * \
                    self.equations[i][entering_var]
                self.equations[i] += self.equations[i][entering_var] * \
                    (- self.equations[row_ind_leaving_var, :])

    def run(self):
        while True:
            self.n_iterations += 1
            entering_var = self.get_entering_var()
            leaving_var = self.get_leaving_var(entering_var)
            row_ind_leaving_var = [i for i in range(len(self.basic_vars)) \
                                          if self.basic_vars[i] == leaving_var][0] + 1

            self.update_basic_and_non_basic_vars(entering_var, leaving_var)
            self.update_equations(entering_var, leaving_var, row_ind_leaving_var)

            if min(self.equations[0, :]) >= 0:
                break

