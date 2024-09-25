#!/usr/bin/env python3
"""Task 1"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """
    Create a TF-IDF embedding matrix.
      - Create an instance of TfidfVectorizer
      - Fit and transform the sentences
      - Get the feature names (words)
    """

    vectorizer = TfidfVectorizer(vocabulary=vocab)
    embeddings = vectorizer.fit_transform(sentences).toarray()
    features = vectorizer.get_feature_names()

    return embeddings, features
