#!/usr/bin/env python3
"""RNN Encoder for Sequence Data"""
import tensorflow as tf
tf.compat.v1.enable_eager_execution()

class RNNEncoder:
    def __init__(self, vocab, embedding, units, batch):
        """ """
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        init = tf.keras.initializers.glorot_uniform()
        self.gru = tf.keras.layers.GRU(units,
                                       recurrent_initializer=init,
                                       return_sequences=True,
                                       return_state=True)
        def initialize_hidden_state(self):
            """ """
            # initialise the hidden state to zeros
             return tf.zeros((self.batch, self.units))
        
        def call(self, x, initial):
            """
            Forward pass through the RNN Encoder."""
            x = self.embedding(x)  # (batch, input_seq_len, embedding_dim)

            outputs, hidden = self.gru(x, initial_state=initial)

            return outputs, hidden
