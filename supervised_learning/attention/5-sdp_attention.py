#!/usr/bin/env python3
"""Scaled dot product attention"""

import numpy as np


def sdp_attention(Q, K, V, mask=None):
    """Calculate the scaled dot product attention"""
    matmul_qk = np.matmul(Q, K.transpose(0, 1, 3, 2))
    dk = K.shape[-1]
    scaled_attention_logits = matmul_qk / np.sqrt(dk)

    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    attention_weights = softmax(scaled_attention_logits)
    output = np.matmul(attention_weights, V)

    return output, attention_weights


def softmax(x):
    """Apply softmax to the last axis"""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)
