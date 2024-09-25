#!/usr/bin/env python3
"""Task 4"""

from gensim.models import FastText


def fasttext_model(sentences, size=100, min_count=5,
                   negative=5, window=5, cbow=True,
                   iterations=5, seed=0, workers=1):
    """
    Creates and trains a Gensim FastText model.
      - sentences: A list of sentences to be trained on.
      - size: The dimensionality of the embedding layer.
      - min_count: The minimum number of occurrences of
        a word for use in training.
      - window: The maximum distance between the current
        and predicted word within a sentence.
      - negative: The size of negative sampling.
      - cbow: A boolean to determine the training type;
        True for CBOW, False for Skip-gram.
      - iterations: The number of iterations to train over.
      - seed: The seed for the random number generator.
      - workers: The number of worker threads to train the model.
      - Returns the trained FastText model.
    """

    sg = 0 if cbow else 1

    model = FastText(
        sentences,
        vector_size=size,
        min_count=min_count,
        window=window,
        negative=negative,
        sg=sg,
        epochs=iterations,
        seed=seed,
        workers=workers
    )

    return model
