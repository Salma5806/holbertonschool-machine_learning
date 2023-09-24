#!/usr/bin/env python3
"""padding"""
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """cnvolve_grayscale"""
    m, h, w, c = images.shape
    kh, kw, c, nc = kernels.shape
    i_step, j_step = stride

    if padding == 'same':
        # solve n+2p-f+1 = n
        p_h = int(np.ceil((i_step*(h-1)-h+kh)/2))
        p_w = int(np.ceil((j_step*(w-1)-w+kw)/2))
    elif padding == 'valid':
        p_h, p_w = 0, 0

    elif type(padding) == tuple:
        p_h, p_w = padding

    output_h = int((h+2*p_h-kh)/i_step+1)
    output_w = int((w+2*p_w-kw)/j_step+1)
    padded_images = np.pad(
        images, ((0, 0), (p_h, p_h), (p_w, p_w), (0, 0)),
        mode='constant')
    output = np.zeros((m, output_h, output_w, nc))
    for i in range(0, output_h):
        x = i*i_step
        for j in range(0, output_w):
            y = j*j_step
            zoom_in = padded_images[:, x:x+kh, y:y+kw, :]
            for k in range(nc):
                kernel = kernels[:, :, :, k]
                product = kernel * zoom_in
                pixel = np.sum(product, axis=(1, 2, 3))
                output[:, i, j, k] = pixel
    return output
