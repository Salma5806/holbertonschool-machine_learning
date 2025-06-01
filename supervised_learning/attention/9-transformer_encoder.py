#!/usr/bin/env python3
"""

"""
import tensorflow as tf
positional_encoding = __import__('4-positional_encoding').positional_encoding
EncoderBlock = __import__('7-transformer_encoder_block').EncoderBlock


import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, LayerNormalization, Dropout


class Encoder(tf.keras.layers.Layer):
   def __init__(self, N, dm, h, hidden, input_vocab, max_seq_len,
                 drop_rate=0.1):
       self.N =N
       self.dm = dm
       self.h = h
       self.hidden = hidden
       self.input_vocab = input_vocab
       self.max_seq_len = max_seq_len
       self.drop_rate = drop_rate
       self.embedding = tf.keras.layers.Embedding(input_dim=input_vocab,
                                                   output_dim=dm)
       self.positional_encoding = positional_encoding(max_seq_len, dm)
       for block in range(N):
        self.blocks = EncoderBlock(dm, h, hidden, drop_rate)
       self.dropout =  tf.keras.layers.Dropout(drop_rate)
       def call(self, x, training, mask=None):
        x = self.embedding(x)
        x = x + self.positional_encoding
        x = self.blocks(x, training, mask)
        x = self.dropout(x, training=training)
        return x
