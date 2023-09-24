#!/usr/bin/env python3
"""pooling module"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """pool for images with multiple channels"""
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    i_step, j_step = stride

    output_h = int((h-kh)/i_step+1)
    output_w = int((w-kw)/j_step+1)
    output = np.zeros((m, output_h, output_w, c))
    for i in range(0, output_h):
        x = i*i_step
        for j in range(0, output_w):
            y = j*j_step
            zoom_in = images[:, x:x+kh, y:y+kw, :]
            if mode == 'max':
                pixel = np.max(zoom_in, axis=(1, 2))
                output[:, i, j, :] = pixel
            elif mode == 'avg':
                pixel = np.mean(zoom_in, axis=(1, 2))
                output[:, i, j, :] = pixel

    return output
