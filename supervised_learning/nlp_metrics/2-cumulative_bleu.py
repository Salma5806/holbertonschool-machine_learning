#!/usr/bin/env python3
"""Task 2"""

from collections import Counter
import math
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


def cumulative_bleu(references, sentence, n):
    """
    Calculates the Cumulative BLEU score for a given sentence and reference(s).
      - references (list): List of reference sentences (list of words).
      - sentence (list): Candidate sentence (list of words) to be evaluated.
      - n (int): Maximum n-gram order for cumulative BLEU score calculation.
      - float: Cumulative BLEU score for the candidate sentence.
      - Initialize weights for different n-gram orders.
      - Iterate over different n-gram orders (up to the specified maximum).
      - Calculate n-grams for the candidate sentence.
      - Iterate over reference sentences.
      - Calculate n-grams for the reference sentence.
      - Update the maximum n-gram counts.
      - Clip the n-gram counts to the maximum counts.
      - Calculate precision for the current n-gram order.
      - Calculate brevity penalty based on sentence length and
        minimum reference length.
      - Calculate cumulative BLEU score using precision values
        and specified n-gram order.
    """
    weights = [1.0/n] * n
    bleu_scores = []

    for i in range(1, 1+n):
        sentence_ngrams = calculate_ngrams(sentence, i)
        sentence_counts = {
            word: sentence_ngrams.count(word) for word in sentence_ngrams}
        max_counts = {}

        for ref in references:
            ref_ngrams = calculate_ngrams(ref, i)
            ref_counts = {word: ref_ngrams.count(word) for word in ref_ngrams}
            for word in ref_counts:
                max_counts[word] = max(
                    max_counts.get(word, 0), ref_counts[word])

        clipped_counts = {word: min(count, max_counts.get(word, 0)) for word,
                          count in sentence_counts.items()}

        bleu_score = sum(clipped_counts.values()) / max(sum(
            sentence_counts.values()), 1)
        bleu_scores.append(bleu_score)

    closest_ref_len = min(len(ref) for ref in references)

    if len(sentence) > closest_ref_len:
        brevity_penalty = 1
    else:
        brevity_penalty = np.exp(1 - closest_ref_len / len(sentence))

    cumulative_bleu_score = brevity_penalty * np.exp(
        sum(w*np.log(s) for w, s in zip(weights, bleu_scores)))

    return cumulative_bleu_score
