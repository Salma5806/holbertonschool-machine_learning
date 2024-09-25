#!/usr/bin/env python3
"""Task 0"""

import numpy as np
import re


def bag_of_words(sentences, vocab=None):
    """
    Creates a bag of words embedding matrix.
      - sentences (list): A list of sentences to analyze.
      - vocab (list, optional): A list of the vocabulary words to
        use for the analysis.
        If None, all words within sentences should be used.
      - embeddings (numpy.ndarray): A 2D array of shape (s, f)
        containing the embeddings, where s is the number of sentences
        and f is the number of features.
      - vocab (list): A list of the features used for embeddings.
      - Tokenize the sentences into words, ignoring punctuation
        (except in numbers) and converting to lowercase.
      - If no vocabulary is provided, create one from the unique
        words in the sentences.
      - Create an empty matrix for the embeddings.
      - Fill in the matrix with the word counts.
    """

    words_list = [re.findall(r'\b\w[\w]*\b', re.sub(
      r"'s\b", '', sentence.lower())) for sentence in sentences]

    if vocab is None:
        vocab = sorted(set(word for words in words_list for word in words))

    embeddings = np.zeros((len(sentences), len(vocab)), dtype=int)

    for i, words in enumerate(words_list):
        for word in words:
            if word in vocab:
                j = vocab.index(word)
                embeddings[i, j] += 1

    return embeddings, vocab
