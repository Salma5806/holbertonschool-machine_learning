#!/usr/bin/env python3
"""Task 8"""

import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """Placeholder"""
    try:
        pi, m, S = initialize(X, k)
    except Exception:
        return None, None, None, None, None

    l_prev = 0
    for i in range(iterations):
        g, log_l = expectation(X, pi, m, S)
        pi, m, S = maximization(X, g)

        if verbose and (i % 10 == 0 or i == iterations - 1):
            print('Log Likelihood after {} iterations: {}'.format(
                i, round(log_l, 5)))

        if abs(log_l - l_prev) <= tol:
            break

        l_prev = log_l

    return pi, m, S, g, log_l
