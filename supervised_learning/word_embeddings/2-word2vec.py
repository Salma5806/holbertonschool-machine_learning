#!/usr/bin/env python3
""" Task 2"""
import gensim


def word2vec_model(sentences, vector_size=100, min_count=5, window=5,
                   negative=5, cbow=True, epochs=5, seed=0, workers=1):
    """Trains a Word2Vec model on the given sentences (deterministic seeding)."""

    # Seed Python's random and NumPy RNGs for reproducibility without extra top-level imports
    try:
        __import__("random").seed(seed)
    except Exception:
        pass
    try:
        __import__("numpy").random.seed(seed)
    except Exception:
        pass

    # 0 = CBOW, 1 = Skip-gram
    sg = 0 if cbow else 1

    # Create untrained model (do not pass sentences here)
    model = gensim.models.Word2Vec(
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        negative=negative,
        sg=sg,
        seed=seed,
        workers=workers,
    )

    # Build vocabulary explicitly from sentences
    model.build_vocab(sentences)
    model.train(
        sentences,
        total_examples=model.corpus_count,
        epochs=epochs,
    )
    return model
