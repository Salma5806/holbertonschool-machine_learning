#!/usr/bin/env python3
import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention
"""
Task RNN Decoder
"""


class RNNDecoder(tf.keras.layers.Layer):
    """ Class to decode for machine translation"""
    def __init__(self, vocab, embedding, units, batch):
        super(RNNDecoder, self).__init__()
        self.units = units
        self.batch = batch
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab, output_dim=embedding)
        self.gru = tf.keras.layers.GRU(units, return_sequences=True, return_state=True,
                       recurrent_initializer='glorot_uniform')
        self.F = tf.keras.layers.Dense(vocab)
        self.attention = SelfAttention(units)
    
    def call(self, x, s_prev, hidden_states):
        """Returns the output word as a one hot vector and
            the new decoder hidden state"""
        x = self.embedding(x)
        context, _ = self.attention(s_prev, hidden_states)
        x = tf.concat([tf.expand_dims(context, 1), x], axis=-1)
        output, s = self.gru(x, initial_state=s_prev)
        y = self.F(tf.reshape(output, (-1, output.shape[2])))
        return y, s
