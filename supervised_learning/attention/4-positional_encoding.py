#!/usr/bin/env python3
import numpy as np
"""positional_encoding.py
Calculates the positional encoding for a transformer"""

def positional_encoding(max_seq_len, dm):
    """"Calculates the positional encoding for a transformer"""
    pe = np.zeros((max_seq_len, dm))
    for i in range(dm):
        for m in range(max_seq_len):
            pe[m, i] = m / np.power(10000, (2 * (i // 2) / dm))
    pe[:, 0::2] = np.sin(pe[:, 0::2])
    pe[:, 1::2] = np.cos(pe[:, 1::2])
    return pe
