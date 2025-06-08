#!/usr/bin/env python3
"""
function that creates masks for training/validation
"""
import tensorflow as tf


def create_padding_mask(seq):
    """
    creates a mask to be applied for a given sequence
    """
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    """(batch_size, 1, 1, seq_len)"""
    return seq[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(size):
    """creates a mask"""
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    """(seq_len, seq_len)"""
    return mask


def create_masks(inputs, target):
    """encoder and decoder masks and look ahead mask"""
    encoder_mask = create_padding_mask(inputs)
    decoder_padding_mask = create_padding_mask(inputs)
    look_ahead_mask = create_look_ahead_mask(tf.shape(target)[1])
    dec_target_padding_mask = create_padding_mask(target)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
    return encoder_mask, combined_mask, decoder_padding_mask
