#!/usr/bin/env python3
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense
SelfAttention = __import__('5-sdp_attention').sdp_attention
"""
Direct application of the
equation
"""
class MultiHeadAttention(Layer):
    def __init__(self, dm, h):
        super(MultiHeadAttention, self).__init__()
        self.h = h  
        self.dm = dm 
        self.depth = dm // h  
        self.Wq = Dense(dm) 
        self.Wk = Dense(dm)  
        self.Wv = Dense(dm) 
        self.linear = Dense(dm) 

    def call(self, Q, K, V, mask):
        batch_size = tf.shape(Q)[0]

        Q = self.Wq(Q) 
        K = self.Wk(K)
        V = self.Wv(V)

        Q = self.split_heads(Q, batch_size) 
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)
        outputs = []
        attention_weights = []
        for i in range(self.h):
            Q_head = Q[:, i, :, :]
            K_head = K[:, i, :, :]
            V_head = V[:, i, :, :]

            output, weights = sdp_attention(Q_head, K_head, V_head, mask)
            outputs.append(output)
            attention_weights.append(weights)

        concat_output = tf.concat(outputs, axis=-1)
        output = self.linear(concat_output)

        return output, attention_weights

    def split_heads(self, x, batch_size):
        """Reshape pour diviser les dernières dimensions en (h, depth)."""
        x = tf.reshape(x, (batch_size, -1, self.h, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])