#!/usr/bin/env python3
"""
positional_encoding.py
Calculates the positional encoding for a transformer
"""

import numpy as np


def positional_encoding(max_seq_len, dm):
    """
    Calculates the positional encoding for a transformer

    Args:
        max_seq_len (int): maximum sequence length
        dm (int): model depth (dimension)

    Returns:
        numpy.ndarray of shape (max_seq_len, dm) with positional encodings
    """
    pe = np.zeros((max_seq_len, dm))
    for i in range(max_seq_len):
        for m in range(dm):
            angle = i / np.power(10000, (2 * (m // 2)) / dm)
            if m % 2 == 0:
                pe[i, m] = np.sin(angle)
            else:
                pe[i, m] = np.cos(angle)
    return pe
