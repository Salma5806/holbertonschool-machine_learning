#!/usr/bin/env python3
import tensorflow as tf
"""
Direct application of the
equation
"""
def sdp_attention(Q, K, V, mask=None):
    """
    Applies scaled dot product attention to the inputs Q, K, and V."""
    Q = tf.convert_to_tensor(Q, dtype='float32')
    K = tf.convert_to_tensor(K, dtype='float32')
    V = tf.convert_to_tensor(V, dtype='float32')
    MatMul = tf.matmul(Q, K, transpose_b=True)
    scale = 1 / tf.math.sqrt(tf.cast(tf.shape(Q)[-1], tf.float32))
    scaled_score = MatMul * scale
    if mask is not None:
        scaled_score = scaled_score + (mask * -1e9)

    weights = tf.nn.softmax(scaled_score, axis=-1)
    output = tf.matmul(weights, V)
    return output, weights