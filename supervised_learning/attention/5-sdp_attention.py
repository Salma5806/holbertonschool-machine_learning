#!/usr/bin/env python3
"""
Defines a function that calculates the scaled dot product attention
"""


import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """
    Calculates the scaled dot product attention

    """
    matmul_qk = tf.matmul(Q, K, transpose_b=True)
    # scale matmul_qk
    dk = tf.cast(tf.shape(K)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    # add mask to scaled tensor
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    # normalize softmax on last axis so all scores add up to 1
    weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    # calculate outputs
    outputs = tf.matmul(weights, V)
    return outputs, weights
