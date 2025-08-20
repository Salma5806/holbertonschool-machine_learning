#!/usr/bin/env python3
"""Task 1"""
from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """Converts a list of sentences into a TF-IDF embedding matrix"""
    vectorizer = TfidfVectorizer(vocabulary=vocab)
    x = vectorizer.fit_transform(sentences)
    embedding = x.toarray()
    features = vectorizer.get_feature_names_out()

    return embedding, features
