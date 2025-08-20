#!/usr/bin/env python3
"""task3"""

import tensorflow as tf


def gensim_to_keras(model):
    """Converts a gensim word2vec model to a keras Embedding layer"""
    vocab_size = len(model.wv)
    embedding_dim = model.vector_size
    embedding_layer = tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        weights=[model.wv.vectors],
        trainable=True
    )
    return embedding_layer
