#!/usr/bin/env python3
"""RNNDecoder class for machine translation
This module implements an RNN decoder for machine translation using TensorFlow.
"""

import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """
    Class to decode for machine translation
    """

    def __init__(self, vocab, embedding, units, batch):
        """
        Class constructor

        Args:
            vocab (int): size of the vocabulary
            embedding (int): dimensionality of the embedding vector
            units (int): number of hidden units in the GRU
            batch (int): batch size (not used directly but can be stored if needed)
        """
        super(RNNDecoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab,
                                                   output_dim=embedding)
        self.gru = tf.keras.layers.GRU(units=units,
                                       return_state=True,
                                       return_sequences=True,
                                       recurrent_initializer="glorot_uniform")
        self.F = tf.keras.layers.Dense(units=vocab)

    def call(self, x, s_prev, hidden_states):
        """
        Performs the decoding step

        Args:
            x: input to the decoder (usually the previous target token)
            s_prev: previous decoder hidden state
            hidden_states: encoder hidden states for attention

        Returns:
            y: output logits
            s: new decoder hidden state
        """
        units = s_prev.get_shape().as_list()[1]
        attention = SelfAttention(units)
        context, weights = attention(s_prev, hidden_states)
        x = self.embedding(x)
        context = tf.expand_dims(context, 1)
        x = tf.concat([context, x], axis=-1)
        y, s = self.gru(x)
        y = tf.reshape(y, (-1, y.shape[2]))
        y = self.F(y)
        return y, s
