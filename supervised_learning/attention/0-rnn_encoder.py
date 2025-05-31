#!/usr/bin/env python3
"""
RNNEncoder module for machine translation"""

import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """
    RNNEncoder class that encodes input sequences using a GRU layer.
    """

    def __init__(self, vocab, embedding, units, batch):
        """
        Class constructor"""
        super(RNNEncoder, self).__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        init = tf.keras.initializers.glorot_uniform()
        self.gru = tf.keras.layers.GRU(units,
                                       recurrent_initializer=init,
                                       return_sequences=True,
                                       return_state=True)

    def initialize_hidden_state(self):
        """
        Initializes the hidden state to a tensor of zeros"""
        return tf.zeros((self.batch, self.units))

    def call(self, x, initial):
        """
        Performs the forward pass of the encoder"""
        x = self.embedding(x)
        outputs, hidden = self.gru(x, initial_state=initial)
        return outputs, hidden
