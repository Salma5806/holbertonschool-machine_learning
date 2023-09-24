#!/usr/bin/env python3
""" Pool backwards propagation """
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs back propagation over a pooling layer of a neural network"""
    def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    m, h_new, w_new, c_new = dA.shape
    kh, kw = kernel_shape
    sh, sw = stride
    m, h_prev, w_prev, c = A_prev.shape

    dA_prev = np.zeros_like(A_prev)

    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_new):
                    if mode == "max":
                        window = A_prev[i, h*sh:h*sh+kh, w*sw:w*sw+kw, c]
                        mask = (window == np.max(window))
                        dA_prev[i, h*sh:h*sh+kh, w*sw:w*sw+kw, c] += mask * dA[i, h, w, c]
                    elif mode == "avg":
                        da = dA[i, h, w, c]
                        average = da / (kh * kw)
                        dA_prev[i, h*sh:h*sh+kh, w*sw:w*sw+kw, c] += np.ones((kh, kw)) * average

    return dA_prev
