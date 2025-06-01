#!/usr/bin/env python3
"""Module implementing the Transformer Encoder."""

import tensorflow as tf
positional_encoding = __import__('4-positional_encoding').positional_encoding
EncoderBlock = __import__('7-transformer_encoder_block').EncoderBlock


class EncoderBlock(tf.keras.layers.Layer):
    """Classe représentant un bloc d'encodeur du Transformer"""
    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """Initialise un bloc encodeur"""
        super().__init__()

        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=h, key_dim=dm // h)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden, activation='relu'),
            tf.keras.layers.Dense(dm)
        ])
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, x, training, mask):
        """Applique le bloc encodeur sur les entrées"""
        attn_output = self.mha(x, x, x, attention_mask=mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.norm1(x + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.norm2(out1 + ffn_output)

        return out2
