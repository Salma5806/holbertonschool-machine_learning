#!/usr/bin/env python3
"""padding"""
import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """cnvolve_grayscale"""
    m, h, w, c = images.shape
    kh, kw, _ = kernel.shape

    sh, sw = stride

    if padding == "valid":
        ph, pw = 0, 0
    elif padding == "same":
        ph = int((((h - 1) * sh + kh - h) / 2) + 1)
        pw = int((((w - 1) * sw + kw - w) / 2) + 1)

    else:
        ph, pw = padding
    oh = int((h - kh + 2 * ph) / sh + 1)
    ow = int((w - kw + 2 * pw) / sw + 1)
    output = np.zeros(shape=(m, oh, ow))
    images_padded = np.pad(
        images,
        [
            (0, 0),
            (ph, ph),
            (pw, pw),
            (0, 0)
        ],
        mode="constant"
    )

    for x in range(oh):
        for y in range(ow):
            output[:, x, y] = np.sum(
                (kernel * images_padded[:, x *
                 sh:x * sh + kh, y * sw:y * sw + kw]),
                axis=(1, 2, 3)
            )

    return output
