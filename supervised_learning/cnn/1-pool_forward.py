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

    h_out = int((h_prev - kh) / sh) + 1
    w_out = int((w_prev - kw) / sw) + 1

    A = np.zeros((m, h_out, w_out, c_prev))

    for i in range(h_out):
        for j in range(w_out):
            if mode == "avg":
                pooled = np.average(A_prev[:, i*sh:j*sh + kh,
                                    j*sw:j*sw + kw, :],
                                    axis=(1, 2))

            else:
                pooled = np.amax(A_prev[:, i*sh:i*sh + kh,
                                 j*sw:j*sw + kw, :],
                                 axis=(1, 2))
            A[:, i, j, :] = pooled

    return A
