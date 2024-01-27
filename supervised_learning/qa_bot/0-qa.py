#!/usr/bin/env python3
"""Task 0"""

import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer


def question_answer(input_question, input_reference):
    """
    This function uses a BERT model to answer questions based
    on a reference text.

    Args:
        input_question (str): The question to be answered.
        input_reference (str): The reference text containing the answer.

    Returns:
        str: The answer to the question.
    """
    # Load the tokenizer and the model
    bert_tokenizer = BertTokenizer.from_pretrained(
        'bert-large-uncased-whole-word-masking-finetuned-squad')
    bert_model = hub.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")

    # Tokenize the question and the reference
    question_tokens = bert_tokenizer.tokenize(input_question)
    reference_tokens = bert_tokenizer.tokenize(input_reference)

    # Prepare the tokens for the model
    bert_tokens = ['[CLS]'] + question_tokens + [
        '[SEP]'] + reference_tokens + ['[SEP]']
    word_ids = bert_tokenizer.convert_tokens_to_ids(bert_tokens)
    mask = [1] * len(word_ids)
    type_ids = [0] * (
        1 + len(question_tokens) + 1) + [1] * (len(reference_tokens) + 1)

    # Convert the inputs to tensors
    tensor_word_ids, tensor_mask, tensor_type_ids = map(
        lambda t: tf.expand_dims(tf.convert_to_tensor(
            t, dtype=tf.int32), 0), (word_ids, mask, type_ids))

    # Get the model outputs
    bert_outputs = bert_model([tensor_word_ids, tensor_mask, tensor_type_ids])

    # Get the start and end of the answer
    answer_start = tf.argmax(bert_outputs[0][0][1:]) + 1
    answer_end = tf.argmax(bert_outputs[1][0][1:]) + 1

    # Get the answer tokens and convert them to a string
    answer_tokens = bert_tokens[answer_start: answer_end + 1]
    bert_answer = bert_tokenizer.convert_tokens_to_string(answer_tokens)

    return bert_answer