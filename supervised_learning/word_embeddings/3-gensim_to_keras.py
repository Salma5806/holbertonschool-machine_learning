#!/usr/bin/env python3
"""Task 3"""

from tensorflow.keras.layers import Embedding
import numpy as np


def gensim_to_keras(model):
    """
    Converts a trained Gensim Word2Vec model to a
    trainable Keras Embedding layer.
      - Get the vocabulary size and embedding dimensions
        from the Gensim model
      - Create an Embedding layer in Keras
    """

    vocab_size, embedding_dim = model.wv.vectors.shape

    keras_embedding_layer = Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        weights=[model.wv.vectors],
        input_length=1,
        trainable=True
    )

    return keras_embedding_layer
