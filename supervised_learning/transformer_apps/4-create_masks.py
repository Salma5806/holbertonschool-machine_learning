#!/usr/bin/env python3


import tensorflow as tf

def create_masks(inputs, target):
    """Generate attention masks for the Transformer model"""

    def padding_mask(seq):
    	""" """
        mask = tf.cast(seq == 0, tf.float32)
        return mask[:, tf.newaxis, tf.newaxis, :]

    def look_ahead_mask(length):
        """ """
        return 1 - tf.linalg.band_part(tf.ones((length, length)), -1, 0)

    encoder_mask = padding_mask(inputs)

    decoder_pad_mask = padding_mask(target)
    future_mask = look_ahead_mask(tf.shape(target)[1])
    combined_mask = tf.maximum(decoder_pad_mask[:, :, :, :tf.shape(target)[1]],
                               future_mask[tf.newaxis, tf.newaxis, :, :])

    decoder_mask = encoder_mask

    return encoder_mask, combined_mask, decoder_mask

