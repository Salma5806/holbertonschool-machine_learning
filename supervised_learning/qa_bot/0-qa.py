#!/usr/bin/env python3
"""Task 0"""

import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer


def question_answer(input_question, input_reference):
    """This function uses a BERT model to answer questions based"""
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    model = hub.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")
    
    reference_tokens = tokenizer.tokenize(reference) + ['[SEP]']
    reference_ids = tokenizer.convert_tokens_to_ids(reference_tokens)
    
    question_tokens = ['[CLS]'] + tokenizer.tokenize(question) + ['[SEP]']
    question_ids = tokenizer.convert_tokens_to_ids(question_tokens)
    
    input_ids = question_ids + reference_ids
    input_mask = [1] * len(input_ids)
    input_types = [0] * len(question_tokens) + [1] * len(reference_tokens)
    
    inputs = [
        tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
        tf.expand_dims(tf.convert_to_tensor(input_mask, dtype=tf.int32), 0),
        tf.expand_dims(tf.convert_to_tensor(input_types, dtype=tf.int32), 0)
    ]
    
    outputs = model(inputs)
    start = tf.argmax(outputs[0][0][1:-1]) + 1
    end = tf.argmax(outputs[1][0][1:-1]) + 1
    
    answer_tokens = (question_tokens + reference_tokens)[start:end+1]
    return tokenizer.convert_tokens_to_string(answer_tokens) if answer_tokens else None
