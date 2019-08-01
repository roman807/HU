#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Roman Moser, 7/21/19

"""
CISC 601 -Scientific Computing II - Assignment 2
Chapra, S. C., & Canale, R. P. (2015). Numerical Methods for Engineers (7th ed.).
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# --------------- Problem 8.3, page 216 ---------------  #
# define functions:
def f(x, pt=3, K_goal=0.05):
    K  = (x / (1 - x)) * np.sqrt((2 * pt) / (2 + x))
    return K - K_goal

def find_root(x_lower, x_upper, x_r, es=0.01, iter_max=1000, method='bisection'):
    i = 0
    ea = es + 1
    while ea > es and i < iter_max:
        x_r_old = x_r
        if method == 'bisection':
            x_r = (x_lower + x_upper) / 2
        elif method == 'false-position':
            x_r = x_upper - (f(x_upper) * (x_lower - x_upper) / (f(x_lower) - f(x_upper))) 
        i += 1
        ea = np.abs((x_r - x_r_old) / x_r) * 100
        test = f(x_lower) * f(x_r)
        if test < 0:
            x_upper = x_r
        elif test > 0:
            x_lower = x_r
        else:
            ea = 0
    return x_r

# find root with false position method:
x_lower, x_upper = -1.9, 0.9
x_r = x_lower
es = 1e-9
iter_max = 1000
method = 'false-position'
res = find_root(x_lower, x_upper, x_r, es=es, iter_max=iter_max, method=method)
print('x found with {} method: '.format(method), res)
print('f(x) with x_result found: ', f(res))

# plot result
x_ = np.linspace(res - 0.5, res + 0.5, 100)
f_x = [f(x) for x in x_]
plt.plot(x_, f_x)
plt.axvline(res, c='red')
plt.title('f(x) in area [x_result - 0.5, x_result + 0.5]')
plt.ylabel('f(x)')
plt.grid()
plt.show()


# ---------- Problem 12.6: naive Gauss elimination (page 331) ---------- #
### algorithm: page 254
def naive_gauss_elimination(A, b):
    # forward elimination:
    n = A.shape[0]
    x = np.zeros([n, 1])
    for k in range(n - 1):
        for i in range(k + 1, n):
            factor = A[i, k] / A[k, k]
            for j in range(k + 1, n):
                A[i, j] = A[i, j] - factor * A[k, j]
            b[i] = b[i] - factor * b[k]
    # back substitution:
    x[n - 1] = b[n - 1] / A[n - 1, n - 1]
    for i in range(n - 2, -1, -1):
        print(i)
        sum_ = b[i]
        for j in range(i + 1, n):
            sum_ -= A[i, j] * x[j]
        x[i] = sum_ / A[i, i]
    return x
    
A = np.array([[-120, 20, 0],
              [80, -80, 0],
              [40, 60, -120]])
b = np.array([-400, 0, -200])
x = naive_gauss_elimination(A.copy(), b.copy())
print('x = \n', x)
print('b = ', np.matmul(A, x))
### --> result inaccurate: why???


# ---------- Problem 2.8: Interest (page 50) ---------- #
def future_value(P, interest, n):
    n_, F = [], []
    for i in range(1, n + 1):
        n_.append(i)
        F.append(P * (1 + interest) ** i)
    df = pd.DataFrame({'n': n_, 'F': F})
    df = df.set_index('n')
    return df

principal = 100000
interest = 0.04
n = 11
print(future_value(principal, interest, n))


# ---------- Problem 2.9: Repayment (page 50) ---------- #
def repayment(P, interest, n):
    n_, A = [], []
    for i in range(1, n + 1):
        n_.append(i)
        A.append(P * (interest * (1 + interest) ** i) / (((1 + interest) ** n) - 1))
    df = pd.DataFrame({'n': n_, 'A': A})
    df = df.set_index('n')
    return df

principal = 55000
interest = 0.066
n = 5
print(repayment(principal, interest, n))


