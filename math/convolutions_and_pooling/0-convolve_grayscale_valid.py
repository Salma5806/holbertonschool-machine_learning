#!/usr/bin/env python3
"""convolve_grayscale_valid"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """cnvolve_grayscale_valid"""
    m, h, w = images.shape
    kh, kw = kernel.shape
    output_h = h - kh + 1  # padding = 0
    output_w = w - kw + 1  # padding = 0
    output = np.zeros((m, output_h, output_w))
    for i in range(output_h):
        for j in range(output_w):
            zoom_in = images[:, i:i+kh, j:j+kw]
            product = kernel * zoom_in
            pixel = np.sum(product, axis=(1, 2))
            output[:, i, j] = pixel
    return output
