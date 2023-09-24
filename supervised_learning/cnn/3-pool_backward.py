#!/usr/bin/env python3
""" Pool backwards propagation """
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs back propagation over a pooling layer of a neural network"""
    m, h_new, w_new, c_new = dA.shape
    kh, kw = kernel_shape
    sh, sw = stride

    dA_prev = np.zeros(A_prev.shape)
    for img in range(m):
        for channel in range(c_new):
            for i in range(h_new):
                for j in range(w_new):
                    if mode == "max":
                        pool = A_prev[img,
                                      i * sh:i * sh + kh,
                                      j * sw:j * sw + kw,
                                      channel]
                        mask = pool == np.max(pool)
                        dA_prev[img,
                                i * sh:i * sh + kh,
                                j * sw:j * sw + kw,
                                channel] += dA[img, i, j, channel] * mask
                    else:
                        d = dA[img, i, j, channel] / (kh * kw)
                        dA_prev[img,
                                i * sh:i * sh + kh,
                                j * sw:j * sw + kw,
                                channel] += d
    return dA_prev
