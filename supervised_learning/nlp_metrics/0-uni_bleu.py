#!/usr/bin/env python3
"""Task 0"""

from collections import Counter
import math


def uni_bleu(references, sentence):
    """
    Calculates the Unigram BLEU score for a
    given sentence and reference(s).
      - references (list): List of reference sentences (list of words).
      - sentence (list): Candidate sentence (list of words) to be evaluated.
      - bleu_score (float): Unigram BLEU score for the candidate sentence.
      - Initialize a list to store precision values for different
        n-gram orders.
      - Iterate over different n-gram orders (unigrams in this case).
      - Initialize counters for n-grams in references and the
        candidate sentence.
      - Update counters with n-grams for references and the candidate sentence.
      - Calculate the common n-grams between the candidate and references.
      - Calculate precision for the current n-gram order.
      - Append precision to the list.
      - Calculate brevity penalty based on sentence length and minimum
        reference length.
      - Calculate BLEU score using precision values and brevity penalty.
      - Return the calculated BLEU score.
    """

    precisions = []
    for i in range(1, 2):
        reference_ngrams = Counter()
        candidate_ngrams = Counter()

        for reference in references:
            reference_ngrams.update(
                zip(*[reference[j:] for j in range(i)]))

        candidate_ngrams.update(
            zip(*[sentence[j:] for j in range(i)]))

        common_ngrams = candidate_ngrams & reference_ngrams
        precision = sum(common_ngrams.values()) / max(
            1, sum(candidate_ngrams.values()))

        precisions.append(precision)

    reference_lengths = [len(ref) for ref in references]
    closest_ref_length = min(
      reference_lengths, key=lambda x: abs(len(sentence) - x))

    if len(sentence) < closest_ref_length:
        brevity_penalty = math.exp(1 - closest_ref_length / len(sentence))
    else:
        brevity_penalty = 1.0

    bleu_score = brevity_penalty * math.exp(sum(
        math.log(p) for p in precisions) / len(precisions))

    return bleu_score
