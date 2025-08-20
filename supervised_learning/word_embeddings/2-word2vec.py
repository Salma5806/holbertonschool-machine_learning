#!/usr/bin/env python3
""" Task 2 """
import gensim


def word2vec_model(sentences, vector_size=100, min_count=5, window=5,
                   negative=5, cbow=True, epochs=5, seed=0, workers=1):
    """Trains a Word2Vec model on the given sentences"""
    sg = 0 if cbow else 1
    model = gensim.models.Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=sg,
        negative=negative,
        seed=seed,
        workers=workers,
        epochs=epochs
    )
    return model
