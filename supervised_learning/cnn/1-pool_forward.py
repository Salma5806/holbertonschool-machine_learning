#!/usr/bin/env python3
""" Convolutional Forward Propagation """
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """ func performs forward propagation over pooling layer """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh = kernel_shape[0]
    kw = kernel_shape[1]
    sh = stride[0]
    sw = stride[1]

    h_out = int(1 + (h_prev - kh) / sh)
    w_out = int(1 + (w_prev - kw) / sw)
    A = np.zeros((m, h_out, w_out, c_prev))

    for i in range(h_out):
        for j in range(w_out):
            h_start = i * sh
            h_end = h_start + kh
            w_start = j * sw
            w_end = w_start + kw
            A_slice = A_prev[:, h_start:h_end, w_start:w_end, :]

            if mode == "max":
                A[:, i, j, :] = np.max(A_slice, axis=(1, 2))
            elif mode == "avg":
                A[:, i, j, :] = np.mean(A_slice, axis=(1, 2))

    return A
