#!/usr/bin/env python3
"""Module implementing the Multi Head Attention layer."""

import tensorflow as tf


MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class DecoderBlock(tf.keras.layers.Layer):
    """
    Transformer decoder block with multi-head attention and
    feed-forward network.
    """

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """Initialise le bloc encodeur"""
        super(DecoderBlock, self).__init__()
        self.dm = dm
        self.h = h
        self.hidden = hidden
        self.drop_rate = drop_rate

        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)

        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """Passe avant du bloc encodeur"""
        output_1, _ = self.mha1(x, x, x, look_ahead_mask)
        output_1 = self.dropout1(output_1, training=training)
        out_1 = self.layernorm1(x + output_1)

        output_2, _ = self.mha2(
            out_1, encoder_output, encoder_output, padding_mask
        )
        output_2 = self.dropout2(output_2, training=training)
        out_2 = self.layernorm2(out_1 + output_2)

        out_2_dense = self.dense_hidden(out_2)
        out_2_dense = self.dense_output(out_2_dense)
        out_3_dense = self.dropout3(out_2_dense, training=training)
        out_3 = self.layernorm3(out_2 + out_3_dense)

        return out_3
