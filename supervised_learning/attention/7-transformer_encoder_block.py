#!/usr/bin/env python3
"""Module implementing the Transformer Encoder Block."""

import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class EncoderBlock(tf.keras.layers.Layer):
    """Transformer encoder block with multi-head attention and feed-forward network"""

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """Initializes the encoder block"""
        super(EncoderBlock, self).__init__()
        self.dm = dm
        self.h = h
        self.hidden = hidden
        self.drop_rate = drop_rate

        self.mha = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask=None):
        """Forward pass for the encoder block"""
        output_1, _ = self.mha(x, x, x, mask)
        output_1 = self.dropout1(output_1, training=training)
        out_1 = self.layernorm1(x + output_1)

        output_2 = self.dense_hidden(out_1)
        output_2 = self.dense_output(output_2)
        output_2 = self.dropout2(output_2, training=training)
        out_2 = self.layernorm2(out_1 + output_2)

        return out_2
