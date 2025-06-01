#!/usr/bin/env python3
import numpy as np
"""positional_encoding.py
Calculates the positional encoding for a transformer"""

def positional_encoding(max_seq_len, dm):
    """"Calculates the positional encoding for a transformer"""
    pos_encoding = np.zeros((max_seq_len, dm))
    for i in range(dm):
        for pos in range(max_seq_len):
            pos_encoding[pos, i] = pos / np.power(10000, (2 * (i // 2) / dm))
    pos_encoding[:, 0::2] = np.sin(pos_encoding[:, 0::2])
    pos_encoding[:, 1::2] = np.cos(pos_encoding[:, 1::2])
    return pos_encoding
