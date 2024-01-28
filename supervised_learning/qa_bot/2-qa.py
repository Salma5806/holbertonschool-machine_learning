#!/usr/bin/env python3
"""Task 2"""

import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer

question_answer = __import__('0-qa').question_answer


def answer_loop(reference):
    """
    This function answers questions from a reference text.

    Args:
        reference (str): The reference text.

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

            # Fetch the answer using the fetch_answer function
            answer = question_answer(question, reference)

            # If the answer is empty, print the error message
            if not answer:
                print("A: Sorry, I do not understand your question.")
            else:
                print("A: {}".format(answer))

        except KeyboardInterrupt:
            break