#!/usr/bin/env python3
"""
Transformer Network: Combines encoder and decoder into a complete Transformer.
"""

import tensorflow as tf
Encoder = __import__('9-transformer_encoder').Encoder
Decoder = __import__('10-transformer_decoder').Decoder


class Transformer(tf.keras.Model):
    """
    Transformer Model

    Combines encoder and decoder into a full transformer architecture for
    sequence-to-sequence tasks like translation or summarization.
    """

    def __init__(self, N, dm, h, hidden, input_vocab,
                 target_vocab, max_seq_input, max_seq_target,
                 drop_rate=0.1):
        """Initializes the Transformer model"""
        super().__init__()
        self.encoder = Encoder(
            N, dm, h, hidden, input_vocab, max_seq_input, drop_rate
        )
        self.decoder = Decoder(
            N, dm, h, hidden, target_vocab, max_seq_target, drop_rate
        )
        self.linear = tf.keras.layers.Dense(target_vocab)

    def call(self, inputs, target, training,
             encoder_mask, look_ahead_mask, decoder_mask):
        """Performs the forward pass of the transformer model"""
        encoder_output = self.encoder(inputs, training, encoder_mask)
        decoder_output = self.decoder(
            target, encoder_output, training, look_ahead_mask, decoder_mask
        )
        final_output = self.linear(decoder_output)
        return final_output
