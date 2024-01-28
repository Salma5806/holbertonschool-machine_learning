#!/usr/bin/env python3
"""Task 3"""

import os
import numpy as np
import tensorflow_hub as hub
from sklearn.metrics.pairwise import cosine_similarity


def semantic_search(corpus_path, sentence):
    """
    Perform semantic search on a corpus of documents.

    Args:
        corpus_path (str): The path to the corpus of reference documents.
        sentence (str): The sentence from which to perform semantic search.

    Returns:
        str: The reference text of the document most similar to sentence.
    """
    # Read all documents from the corpus into a list
    documents = []
    for filename in os.listdir(corpus_path):
        with open(os.path.join(corpus_path, filename), 'r') as f:
            documents.append(f.read())

    # Add the input sentence to the list of documents
    documents.append(sentence)

    # Load the Universal Sentence Encoder
    model = hub.load(
        "https://tfhub.dev/google/universal-sentence-encoder-large/5")

    # Create embeddings for the documents and the sentence
    embeddings = model(documents)

    # Compute the cosine similarity between the sentence
    # embedding and each document embedding
    cosine_similarities = cosine_similarity(embeddings[-1:], embeddings[:-1])

    # Find the index of the most similar document
    most_similar_index = cosine_similarities.argmax()

    # Return the text of the most similar document
    return documents[most_similar_index]