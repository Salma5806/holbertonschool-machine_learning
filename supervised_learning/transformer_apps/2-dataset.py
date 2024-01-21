#!/usr/bin/env python3
"""Task 2"""

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np


class Dataset:
    """
    This class handles the loading, tokenization, encoding, and mapping of the
    ted_hrlr_translate/pt_to_en dataset from TensorFlow Datasets.
    """

    def __init__(self):
        """
        Constructor method. Loads the train and validation splits of the
        dataset, tokenizes them, encodes them, and maps the encoding
        function to the data.
        """
        # Load the training split of the dataset
        self.data_train = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split='train', as_supervised=True)

        # Load the validation split of the dataset
        self.data_valid = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split='validation', as_supervised=True)

        # Tokenize the training dataset
        self.toke_pt, self.toke_en = self.tokenize_dataset(self.data_train)

        # Map the encoding function to the training data
        self.data_train = self.data_train.map(self.tf_encode)

        # Map the encoding function to the validation data
        self.data_valid = self.data_valid.map(self.tf_encode)

    def tokenize_dataset(self, data):
        """
        Tokenizes the dataset.

        Parameters:
        data (tf.data.Dataset): The dataset to tokenize.

        Returns:
        toke_pt (tfds.deprecated.text.SubwordTextEncoder):
        The tokenizer for Portuguese text.
        toke_en (tfds.deprecated.text.SubwordTextEncoder):
        The tokenizer for English text.
        """
        # Build a tokenizer for the Portuguese text
        toke_pt = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            (pt.numpy() for pt, en in data), target_vocab_size=2**15)

        # Build a tokenizer for the English text
        toke_en = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            (en.numpy() for pt, en in data), target_vocab_size=2**15)

        return toke_pt, toke_en

    def encode(self, pt, en):
        """
        Encodes a translation pair into tokens.

        Parameters:
        pt (tf.Tensor): The Portuguese text.
        en (tf.Tensor): The English text.

        Returns:
        pt_tokens (List[int]): The tokenized Portuguese text.
        en_tokens (List[int]): The tokenized English text.
        """
        # Tokenize the Portuguese text
        pt_tokens = [self.toke_pt.vocab_size] + self.toke_pt.encode(
            pt.numpy()) + [self.toke_pt.vocab_size+1]

        # Tokenize the English text
        en_tokens = [self.toke_en.vocab_size] + self.toke_en.encode(
            en.numpy()) + [self.toke_en.vocab_size+1]

        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """
        Wraps the encode method in a TensorFlow py_function.

        Parameters:
        pt (tf.Tensor): The Portuguese text.
        en (tf.Tensor): The English text.

        Returns:
        pt (tf.Tensor): The tokenized Portuguese text.
        en (tf.Tensor): The tokenized English text.
        """
        # Wrap the encode method in a TensorFlow py_function
        pt, en = tf.py_function(self.encode, [pt, en], [tf.int64, tf.int64])

        # Set the shape of the tensors
        pt.set_shape([None])
        en.set_shape([None])

        return pt, en