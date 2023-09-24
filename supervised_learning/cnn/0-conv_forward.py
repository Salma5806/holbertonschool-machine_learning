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
        pad_h = max(0, (h_out - 1) * sh + kh - h_prev)
        pad_w = max(0, (w_out - 1) * sw + kw - w_prev)
        A_prev = np.pad(A_prev, ((0, 0), (pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2), (0, 0)), mode='constant')
    else:
        h_out = int(np.floor(float(h_prev - kh) / float(sh))) + 1
        w_out = int(np.floor(float(w_prev - kw) / float(sw))) + 1

    Z = np.zeros((m, h_out, w_out, c_new))
    for i in range(h_out):
        for j in range(w_out):
            h_start = i * sh
            h_end = h_start + kh
            w_start = j * sw
            w_end = w_start + kw
            A_slice = A_prev[:, h_start:h_end, w_start:w_end, :]
            Z[:, i, j, :] = np.sum(np.multiply(A_slice, W), axis=(1, 2, 3)) + b

    A = activation(Z)

    return A
