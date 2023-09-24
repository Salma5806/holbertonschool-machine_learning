#!/usr/bin/env python3
"""padding"""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """cnvolve_grayscale"""
    m, h, w = images.shape
    kh, kw = kernel.shape
    i_step, j_step = stride

    if padding == 'same':
        # solve n+2p-f+1 = n
        p_h = int(kh + (i_step*(h-1)-h)/2) 
        p_w = int((kw + (j_step*(w-1)-w)/2))
    elif padding == 'valid':
        p_h, p_w = 0, 0

    elif type(padding) == tuple:
        p_h, p_w = padding

    output_h = int((h+2*p_h-kh)/i_step+1)
    output_w = int((w+2*p_w-kw)/j_step+1)
    padded_images = np.pad(
        images, ((0, 0), (p_h, p_h), (p_w, p_w)),
        mode='constant')
    output = np.zeros((m, output_h, output_w))
    for i in range(0, output_h):
        x = i*i_step
        for j in range(0, output_w):
            y = j*j_step
            zoom_in = padded_images[:, x:x+kh, y:y+kw]
            product = kernel * zoom_in
            pixel = np.sum(product, axis=(1, 2))
            output[:, i, j] = pixel

    return output
