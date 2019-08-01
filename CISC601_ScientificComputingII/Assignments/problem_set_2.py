#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Roman Moser, 8/1/19

"""
CISC 601 -Scientific Computing II - Problem Set 2
Chapra, S. C., & Canale, R. P. (2015). Numerical Methods for Engineers (7th ed.).
Optimization problems from chapter 16, page 431
"""

import matplotlib.pyplot as plt
import numpy as np

# ---------- Problem 16.1, page 431 ---------- #
def h_constrained(r):
    """ constraint (V=0.5) """
    return 0.5 / (np.pi * r**2)

def h(r, A):
    """ calculate h based on r and A """
    return 0.5 * ((A / (np.pi * r)) - r)

def A(r, h):
    """The Area to minimize (objective function) """
    return np.pi * r * (2 * h + r)

# graphical method: estimate solution
r_ = np.linspace(0.1, 2, 100)
h_ = [h_constrained(r1) for r1 in r_]
plt.plot(r_, h_, c='black', linewidth=2.5, label='V=0.5')
for A_ in [0, 2, 4, 6, 10]:
    h_ = [h(r1, A_) for r1 in r_] 
    plt.plot(r_, h_, label='A={}'.format(A_))
plt.xlabel('r')
plt.ylabel('h')
plt.legend()
plt.title('Graphical method')
plt.show()

# graphical method: zoom in to estimate solution
r_ = np.linspace(0.25, 0.8, 100)
h_ = [h_constrained(r1) for r1 in r_]
plt.plot(r_, h_, c='black', linewidth=2.5, label='V=0.5')
for A_ in [2, 2.8, 4]:
    h_ = [h(r1, A_) for r1 in r_] 
    plt.plot(r_, h_, label='A={}'.format(A_))
r_est = 0.55
h_est = 0.54
plt.scatter(r_est, h_est, s=100, c='red', marker='x')
plt.xlabel('r')
plt.ylabel('h')
plt.legend()
plt.grid()
plt.title('Graphical method (zoom in)')
plt.show()

print('estimated minimum with graphical method: r={}, h={}, A={}'.format(
        r_est,
        h_est,
        np.round(A(r_est, h_est), 3)))

# Generalized Reduced Gradient (GRG) method
# reduce problem to unconstrained optimization problem
def A_GRG(r):
    """Area to minimize (objective function) """
    return np.pi * r * (2 * h_constrained(r) + r)

r_ = np.linspace(0.25, 0.8, 100)
A_ = [A_GRG(r1) for r1 in r_]
plt.plot(r_, A_, c='black', linewidth=2.5, label='V=0.5')
r_est = 0.545
A_est = 2.77
plt.scatter(r_est, A_est, s=100, c='red', marker='x')
plt.xlabel('r')
plt.ylabel('h')
plt.legend()
plt.title('Generalized Reduced Gradient method')
plt.grid()
plt.show()
print('estimated minimum with GRG graphic: r={}, h={}, A={}\n'.format(
        r_est,
        np.round(h(r_est, A_est), 3),
        A_est))

# Analytical solution with GRG:
# d(A_GRG)/dr = 0 leads to:
r1 = (1 / (2 * np.pi)) ** (1 / 3)
print('analytical solution with GRG graphic: r={}, h={}, A={}'.format(
        np.round(r1, 3),
        np.round(h_constrained(r1), 3),
        np.round(A_GRG(r1), 3)))


# ---------- Problem 16.8, page 432 ---------- #
# Minimize cost as function of conversion rate x_A
def cost(x_A, C=1):
    return C * ((1 / (1 - x_A) ** 2) ** 0.6 + 5 * (1 / x_A) ** 0.6)

# Graphical estimation:
x = np.linspace(0.01, 0.99, 100)
c = [cost(x_) for x_ in x]
x_est = 0.5
plt.scatter(0.5, 10, marker='x', color='red')
plt.plot(x, c)
plt.title('cost as a function of conversion x_A')
plt.xlabel('x_A')
plt.ylabel('cost')
plt.show()

print('estimated minimum graphical method: x_A={}, cost={}'.format(
        x_est,
        np.round(cost(x_est), 5)))

# Golden section search:
def golden_section_search(x_low, x_high, max_i, es):
    R = (5**0.5 - 1) / 2
    x_l, x_u = x_low, x_high
    i = 1
    d = R * (x_u - x_l)
    x_1, x_2 = x_l + d, x_u - d
    f1, f2 = cost(x_1), cost(x_2)
    if f1 < f2:
        x_opt = x_1
    else:
        x_opt = x_2
    while True:
        d = R * d
        x_int = x_u - x_l
        if f1 < f2:
            x_l, x_2 = x_2, x_1
            x_1 = x_l + d
            f2, f1 = f1, cost(x_1)
        else:
            x_u, x_1 = x_1, x_2
            x_2 = x_u - d
            f1, f2 = f2, cost(x_2)
        i += 1
        if f1 < f2:
            x_opt = x_1
        else:
            x_opt = x_2
        if x_opt != 0:
            ea = (1 - R) * abs(x_int / x_opt) * 100
        if ea <= es or i >= max_i:
            return x_opt

x_low = 0.01
x_high = 0.99
max_i = 10000
es = 0.00001
x_final = golden_section_search(x_low, x_high, max_i, es)

print('estimated minimum golden section search: x_A={}, cost={}'.format(
        np.round(x_final, 5),
        np.round(cost(x_final), 5)))

