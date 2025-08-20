#!/usr/bin/env python3
""" Task 0"""
from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(sentences, vocab=None):
    """Converts a list of sentences into a bag-of-words embedding matrix"""
    vectorizer = CountVectorizer(vocabulary=vocab)
    x = vectorizer.fit_transform(sentences)
    embedding = x.toarray()
    features = vectorizer.get_feature_names_out()

    return embedding, features
