#!/usr/bin/env python3
""" Convolution of a layer """
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """ performs forward propagation over a convolutional
        layer of a neural network """
    m, h_prev, w_prev, c_prev = A.shape
    kh, kw, _, c_new = W.shape
    sh, sw = stride

    if padding == 'valid':
        ph, pw = 0, 0
    else:
        ph = int(np.ceil((h_prev * sh - sh + kh - h_prev) / 2))
        pw = int(np.ceil((w_prev * sw - sw + kw - w_prev) / 2))
    h_new = int((h_prev + 2 * ph - kh) / sh) + 1
    w_new = int((w_prev + 2 * pw - kw) / sw) + 1
    npad = ((0, 0), (ph, ph), (pw, pw), (0, 0))
    A_prev_padded = np.pad(
        A,
        pad_width=npad,
        mode='constant',
        constant_values=0)
    output = np.zeros((m, h_new, w_new, c_new))

    for i in range(m):
        for j in range(h_new):
            for k in range(w_new):
                for x in range(c_new):
                    output[i, j, k, x] = np.sum(
                        A_prev_padded[i, j * sh: j * sh + kh,
                                      k * sw: k * sw + kw, :]
                        * W[:, :, :, x])
    A = activation(output + b)
    return A
