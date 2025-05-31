#!/usr/bin/env python3
import numpy as np
"""positional_encoding.py
Calculates the positional encoding for a transformer"""

def positional_encoding(max_seq_len, dm):
    """"Calculates the positional encoding for a transformer"""
    pe = np.zeros((max_seq_len, dm))
    for i in range(max_seq_len):
        for m in range(dm):
            if int(m) % 2 == 0:
                pe[i,  m] = np.sin(i / (10000 ** (m/ dm)))
            else:
                pe[i, m] = np.cos(i / (10000 ** (m / dm)))
    return pe