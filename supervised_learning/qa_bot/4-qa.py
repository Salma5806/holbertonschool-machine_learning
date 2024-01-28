#!/usr/bin/env python3
"""Task 4"""

import os
import numpy as np
import tensorflow_hub as hub
from sklearn.metrics.pairwise import cosine_similarity

question_answer = __import__('0-qa').question_answer
semantic_search = __import__('3-semantic_search').semantic_search


def question_answer(corpus_path):
    """
    This function answers questions from multiple reference texts.

    Args:
        corpus_path (str): The path to the corpus of reference documents.

    Returns:
        None
    """
    # List of exit commands
    exit_commands = ['exit', 'quit', 'goodbye', 'bye']

    while True:
        try:
            # Get the question from the user
            question = input("Q: ")

            # If the user types an exit command, break the loop
            if question.lower() in exit_commands:
                print("A: Goodbye")
                break

            # Find the most similar document to the question
            similar_document = semantic_search(corpus_path, question)

            # Fetch the answer using the fetch_answer function
            answer = question_answer(question, similar_document)

            # If the answer is empty, print the error message
            if not answer:
                print("A: Sorry, I do not understand your question.")
            else:
                print("A: {}".format(answer))

        except KeyboardInterrupt:
            break