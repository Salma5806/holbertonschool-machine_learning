#!/usr/bin/env python3
"""Module implementing the Multi Head Attention layer."""
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').multihead_attention

class EncoderBlock(tf.keras.layers.Layer):
    """Bloc d'encodage pour un transformeur, avec attention multi-têtes et réseau feed-forward."""

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """Initialise le bloc encodeur"""
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
        """Passe avant du bloc encodeur"""
        output_1, _ = self.mha(x, x, x, mask)
        output_1 = self.dropout1(output_1, training=training)
        out_1 = self.layernorm1(x + output_1)
        output_2 = self.dense_hidden(out_1)
        output_2 = self.dense_output(output_2)
        output_2 = self.dropout2(output_2, training=training)
        out_2 = self.layernorm2(out_1 + output_2)

        return out_2