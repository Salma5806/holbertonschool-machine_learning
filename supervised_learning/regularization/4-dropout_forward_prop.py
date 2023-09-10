#!/usr/bin/env python3
"""
Defines function that conducts forward propagation using Dropout
"""


import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """Conducts forward propagation using Dropout"""
    cache = {}
    cache['A0'] = X
    m = X.shape[1]

    for i in range(1, L + 1):
        W = weights['W' + str(i)]
        b = weights['b' + str(i)]
        A_prev = cache['A' + str(i - 1)]

        Z = np.dot(W, A_prev) + b
        if i == L:
            A = np.exp(Z) / np.sum(np.exp(Z), axis=0)
        else:
            A = np.tanh(Z)
            D = np.random.rand(A.shape[0], A.shape[1])
            D = (D < keep_prob).astype(int)
            A *= D
            A /= keep_prob
            cache['D' + str(i)] = D

        cache['A' + str(i)] = A

    return cache
