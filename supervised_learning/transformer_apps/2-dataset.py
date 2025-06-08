#!/usr/bin/env python3

"""task 2"""

import tensorflow as tf
import tensorflow_datasets as tfds
from transformers import AutoTokenizer


class Dataset:
    """
    This class loads and preps a dataset for machine translation
    """

    def __init__(self):
        """
        Class constructor
        """
        self.data_train = tfds.load('ted_hrlr_translate/pt_to_en',
                                   split='train',
                                   as_supervised=True)
        self.data_valid = tfds.load('ted_hrlr_translate/pt_to_en',
                                   split='validation',
                                   as_supervised=True)

        self.tokenizer_pt = AutoTokenizer.from_pretrained(
            "neuralmind/bert-base-portuguese-cased")
        self.tokenizer_en = AutoTokenizer.from_pretrained(
            "bert-base-uncased")

    def encode(self, pt, en):
        """
        Encodes a translation into tokens (list of ints)
        """
        pt_str = pt.numpy().decode('utf-8')
        en_str = en.numpy().decode('utf-8')

        pt_tokens = self.tokenizer_pt.encode(pt_str)
        en_tokens = self.tokenizer_en.encode(en_str)
        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """
        Encodes a translation pair into tokens using tf.py_function to work with tf.data pipeline
        """
        pt_encoded, en_encoded = tf.py_function(
            func=self.encode,
            inp=[pt, en],
            Tout=[tf.int32, tf.int32])

        pt_encoded.set_shape([None])
        en_encoded.set_shape([None])

        return pt_encoded, en_encoded
