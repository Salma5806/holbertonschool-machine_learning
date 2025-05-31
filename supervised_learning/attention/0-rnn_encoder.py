#!/usr/bin/env python3
"""RNN Encoder for Sequence Data"""
import tensorflow as tf

class RNNEncoder:
    def __init__(self, vocab, embedding, units, batch):
        """
        Initialize the RNN Encoder.

        Args:
            vocab (int): Size of the vocabulary.
            embedding (int): Dimension of the embedding vector.
            units (int): Number of units in the RNN layer.
            batch (int): Batch size for training.
        """
        self.vocab = vocab
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.units = units
        self.batch = batch
        self.gru = tf.keras.layers.GRU(
            units,
            return_sequences=True,
            return_state=True,    
            recurrent_initializer='glorot_uniform'
        )
        def initialize_hidden_state(self):
            # initialise the hidden state to zeros
            return tf.zeros((self.batch, self.units))
        
        def call(self, x, initial):
            """
            Forward pass through the RNN Encoder."""
            x = self.embedding(x)
            output, hidden = self.gru(x, initial_state=initial)
            return output, hidden


