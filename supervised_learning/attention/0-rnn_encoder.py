#!/usr/bin/env python3
import tensorflow as tf
"""
task 0
"""


class RNNEncoder(tf.keras.layers.Layer):
    """
    Class to encode for machine translation
    """
    def __init__(self, vocab, embedding, units, batch):
        """
        Class constructor
        """
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
        Initializes the hidden states for the RNN cell to a tensor of zeros
        """
        return tf.zeros((self.batch, self.units))

    def call(self, x, initial):
        """
        Calls the encoder with given input to encoder layer and returns output
        """
        x = self.embedding(x)
        outputs, hidden = self.gru(x, initial_state=initial)
        return outputs, hidden
