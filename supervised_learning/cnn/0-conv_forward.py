#!/usr/bin/env python3
""" Convolution of a layer """
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """ performs forward propagation over a convolutional
        layer of a neural network """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride
    if padding == "same":
        h_out = int(np.ceil(float(h_prev) / float(sh)))
        w_out = int(np.ceil(float(w_prev) / float(sw)))
        pad_h = max((h_out - 1) * sh + kh - h_prev, 0)
        pad_w = max((w_out - 1) * sw + kw - w_prev, 0)
        A_prev = np.pad(A_prev, ((0, 0), (pad_h // 2, (pad_h + 1) // 2), (pad_w // 2, (pad_w + 1) // 2), (0, 0)), mode='constant')
    else:
        h_out = int(np.floor(float(h_prev - kh + 1) / float(sh)))
        w_out = int(np.floor(float(w_prev - kw + 1) / float(sw)))
    Z = np.zeros((m, h_out, w_out, c_new))
    for i in range(h_out):
        for j in range(w_out):
            for k in range(c_new):
                h_start = i * sh
                h_end = h_start + kh
                w_start = j * sw
                w_end = w_start + kw

                A_slice = A_prev[:, h_start:h_end, w_start:w_end, :]
                Z[:, i, j, k] = np.sum(A_slice * W[:, :, :, k]) + b[:, :, :, k]
    if activation == "relu":
        A = np.maximum(0, Z)
    elif activation == "sigmoid":
        A = 1 / (1 + np.exp(-Z))
    elif activation == "tanh":
        A = np.tanh(Z)
    else:
        A = Z

    return A
