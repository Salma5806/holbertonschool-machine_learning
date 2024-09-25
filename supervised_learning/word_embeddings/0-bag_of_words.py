#!/usr/bin/env python3
"""Task 0"""

import numpy as np
import re


def bag_of_words(sentence):
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

    processed_sentence = re.sub(r"\b(\w+)'s\b", r"\1", sentence.lower())
    return re.findall(r'\w+', processed_sentence)


def bag_of_words(sentences, vocab=None):
    """
    badding word
    """
    preprocessed_sentences = [preprocess_sentence(sentence) 
                        for sentence in sentences]
    if vocab is None:
        all_words = [word for sentence in preprocessed_sentences
                    for word in sentence]
        vocab = sorted(set(all_words))
    word_to_index = {word: idx for idx, word in enumerate(vocab)}
    embeddings = np.zeros((len(sentences), len(vocab)), dtype=int)
    for i, sentence in enumerate(preprocessed_sentences):
        for word in sentence:
            if word in word_to_index:
                embeddings[i, word_to_index[word]] += 1
    return embeddings, vocab