#!/usr/bin/env python3
""" Convolution back propagation """
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    Performs back propagation over a 
    convolutional layer of a neural network
    """
    m, h_new, w_new, c_new = dZ.shape
    m, h_new, w_new, c_new = dZ.shape
    h_prev, w_prev, c_prev = A_prev.shape[1:]
    kh, kw = W.shape[:2]
    sh, sw = stride

    if padding == "valid":
        ph = 0
        pw = 0
    else:
        ph = int((h_new * sh - h_prev + kh - 1) / 2)
        pw = int((w_new * sw - w_prev + kw - 1) / 2)

    padded = np.pad(A_prev,
                    ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                    'constant',
                    constant_values=0)
    dA_prev = np.zeros(padded.shape)
    dW = np.zeros(W.shape)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    for img in range(m):
        for channel in range(c_new):
            for row in range(h_new):
                for col in range(w_new):
                    grad = W[:, :, :, channel] * dZ[img, row, col, channel]
                    dA_prev[img,
                            row * sh:row * sh + kh,
                            col * sw:col * sw + kw,
                            :] += grad

                    grad = padded[img,
                                  row * sh:row * sh + kh,
                                  col * sw:col * sw + kw,
                                  :] * dZ[img, row, col, channel]
                    dW[:, :, :, channel] += grad

    dA_prev = dA_prev[:, ph:dA_prev.shape[1] - ph, pw:dA_prev.shape[2] - pw, :]

    return dA_prev, dW, db
