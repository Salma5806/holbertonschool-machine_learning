#!/usr/bin/env python3
"""
Creates, builds, and trains a Gensim Word2Vec model.
"""

from gensim.models import Word2Vec


def word2vec_model(
    sentences,
    vector_size=100,
    min_count=5,
    window=5,
    negative=5,
    cbow=True,
    epochs=5,
    seed=0,
    workers=1,
):
    """Trains and returns a Word2Vec model"""
    model = Word2Vec(
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        negative=negative,
        sg=0 if cbow else 1,
        seed=seed,
        workers=workers,
    )
    model.build_vocab(sentences)
    model.train(
        sentences,
        total_examples=model.corpus_count,
        epochs=epochs,
    )

    return model
