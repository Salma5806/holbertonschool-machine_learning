#!/usr/bin/env python3
"""Task 1"""

import numpy as np


def calculate_ngrams(tokens, n):
    """
    Helper function to calculate n-grams from a list of tokens.
      - tokens (list): List of tokens (words).
      - n (int): Size of the n-gram.
      - list: List of n-grams.
      - Generate n-grams using a sliding window of size 'n'.
    """

    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def ngram_bleu(references, hypothesis, n):
    """
    Calculates the n-gram BLEU score for a sentence.
      - references (list): List of reference translations.
        Each reference translation is a list of words.
      - hypothesis (list): List containing the model proposed sentence.
      - n (int): Size of the n-gram to use for evaluation.
      - float: The n-gram BLEU score.
      - Calculate n-grams for the model proposed sentence.
      - Initialize a dictionary to store maximum n-gram counts for each n-gram.
      - Iterate over reference translations.
      - Calculate n-grams for the reference translation.
      - Update the maximum n-gram counts.
      - Clip the n-gram counts to the maximum counts.
      - Calculate precision for the n-gram order.
      - Calculate brevity penalty based on sentence length and minimum
        reference length.
      - Calculate the final BLEU score.
    """

    hypothesis_ngrams = calculate_ngrams(hypothesis, n)
    hypothesis_counts = {
        word: hypothesis_ngrams.count(word) for word in hypothesis_ngrams}

    max_counts = {}

    for ref in references:
        ref_ngrams = calculate_ngrams(ref, n)
        ref_counts = {word: ref_ngrams.count(word) for word in ref_ngrams}
        for word in ref_counts:
            max_counts[word] = max(max_counts.get(word, 0), ref_counts[word])

    clipped_counts = {word: min(count, max_counts.get(word, 0)) for word,
                      count in hypothesis_counts.items()}

    bleu_score = sum(clipped_counts.values()) / max(sum(
        hypothesis_counts.values()), 1)

    closest_ref_len = min(len(ref) for ref in references)

    brevity_penalty = 1 if len(
        hypothesis) > closest_ref_len else np.exp(
            1 - closest_ref_len / len(hypothesis))

    bleu_score = brevity_penalty * bleu_score
    return bleu_score
