#!/usr/bin/env python3

"""task 2"""

import tensorflow as tf
import tensorflow_datasets as tfds
import transformers


class Dataset:
    """
    This class loads and preps a dataset for machine translation
    """

    def __init__(self):
        """
        Class constructor
        """
        # Chargement du dataset brut
        raw_train = tfds.load('ted_hrlr_translate/pt_to_en',
                              split='train',
                              as_supervised=True)
        raw_valid = tfds.load('ted_hrlr_translate/pt_to_en',
                              split='validation',
                              as_supervised=True)

        # Création des tokenizers
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(raw_train)

        # Tokenisation et encodage des datasets (avec tf_encode)
        self.data_train = raw_train.map(self.tf_encode,
                                       num_parallel_calls=tf.data.AUTOTUNE)
        self.data_valid = raw_valid.map(self.tf_encode,
                                       num_parallel_calls=tf.data.AUTOTUNE)

    def tokenize_dataset(self, data):
        """Creates sub-word tokenizers for our dataset"""
        tokenizer_pt = transformers.AutoTokenizer.from_pretrained(
            "neuralmind/bert-base-portuguese-cased")
        tokenizer_en = transformers.AutoTokenizer.from_pretrained(
            "bert-base-uncased")

        def iterate_pt():
            for pt, _ in data:
                yield pt.numpy().decode('utf-8')

        def iterate_en():
            for _, en in data:
                yield en.numpy().decode('utf-8')

        # Entraînement des tokenizers sur les données (vocab_size = 8192)
        tokenizer_pt = tokenizer_pt.train_new_from_iterator(
            iterate_pt(), vocab_size=2**13)
        tokenizer_en = tokenizer_en.train_new_from_iterator(
            iterate_en(), vocab_size=2**13)

        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """
        Encodes a translation into tokens
        """
        pt_tokens = [self.tokenizer_pt.vocab_size] + \
            [int(t) for t in self.tokenizer_pt.encode(pt.numpy().decode('utf-8'),
                                                     add_special_tokens=False)] + \
            [self.tokenizer_pt.vocab_size + 1]
        en_tokens = [self.tokenizer_en.vocab_size] + \
            [int(t) for t in self.tokenizer_en.encode(en.numpy().decode('utf-8'),
                                                     add_special_tokens=False)] + \
            [self.tokenizer_en.vocab_size + 1]
        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """
        Acts as a TensorFlow wrapper for the encode instance method
        """
        pt, en = tf.py_function(func=self.encode,
                                inp=[pt, en],
                                Tout=[tf.int64, tf.int64])
        pt.set_shape([None])
        en.set_shape([None])
        return pt, en
