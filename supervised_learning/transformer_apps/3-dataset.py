#!/usr/bin/env python3

"""task 3"""

import tensorflow as tf
import tensorflow_datasets as tfds
import transformers


class Dataset:
    """
    This class loads and preps a dataset for machine translation
    """

    def __init__(self, batch_size, max_len):
        """
        Class constructor
        """
        raw_train = tfds.load('ted_hrlr_translate/pt_to_en',
                             split='train',
                             as_supervised=True)
        raw_valid = tfds.load('ted_hrlr_translate/pt_to_en',
                             split='validation',
                             as_supervised=True)

        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(raw_train)

        train_encoded = raw_train.map(self.tf_encode,
                                     num_parallel_calls=tf.data.AUTOTUNE)
        valid_encoded = raw_valid.map(self.tf_encode,
                                     num_parallel_calls=tf.data.AUTOTUNE)

        def filter_max_len(pt, en):
            return tf.logical_and(tf.size(pt) <= max_len,
                                  tf.size(en) <= max_len)

        self.data_train = (train_encoded
                           .filter(filter_max_len)
                           .cache()
                           .shuffle(buffer_size=20000)
                           .padded_batch(batch_size, padded_shapes=([None], [None]))
                           .prefetch(tf.data.AUTOTUNE))

        self.data_valid = (valid_encoded
                           .filter(filter_max_len)
                           .padded_batch(batch_size, padded_shapes=([None], [None])))

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
