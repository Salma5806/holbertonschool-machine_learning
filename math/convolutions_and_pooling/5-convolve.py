#!/usr/bin/env python3
"""padding"""
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """cnvolve_grayscale"""
    m, h, w, c = images.shape
    kh, kw, c, nc = kernels.shape
    sh, sw = stride

    if isinstance(padding, tuple):
        ph, pw = padding
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph = int(np.ceil((sh * (h - 1) - h + kh) / 2))
        pw = int(np.ceil((sw * (w - 1) - w + kw) / 2))
    output_h = int((h + 2 * ph - kh) / sh + 1)
    output_w = int((w + 2 * pw - kw) / sw + 1)
    npad = ((0, 0), (ph, ph), (pw, pw), (0, 0))
    images = np.pad(images, pad_width=npad, mode='constant')
    output = np.zeros((m, output_h, output_w, nc))
    for i in range(0, output_h):
        x = i * sh
        for j in range(0, output_w):
            y = j * sw
            for k in range(nc):
                output[:, i, j, k] = np.sum(
                    images[:, x:x + kh, y:y + kw] * kernels[:, :, :, k],
                    axis=(1, 2, 3))
    return output
