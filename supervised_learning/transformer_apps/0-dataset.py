#!/usr/bin/env python3
"""Task 0"""

import tensorflow_datasets as tfds
from transformers import AutoTokenizer


class Dataset:
    """
    This class handles the loading and tokenization of the
    ted_hrlr_translate/pt_to_en dataset from TensorFlow Datasets.
    """

    def __init__(self):
        """
        Constructor method. Loads the train and
        validation splits of the dataset and tokenizes them.
        """
        self.data_train = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='train', as_supervised=True)
        self.data_valid = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='validation', as_supervised=True)

        self.tokenizer_pt = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
        self.tokenizer_en = AutoTokenizer.from_pretrained("bert-base-uncased")

    def tokenize_dataset(self, data):
        """Tokenizes the dataset"""
       	tokenized_pt, tokenized_en = [], []
        for pt, en in data:
            pt_str = pt.numpy().decode('utf-8')
            en_str = en.numpy().decode('utf-8')
            tokenized_pt.append(self.tokenizer_pt.encode(pt_str))
            tokenized_en.append(self.tokenizer_en.encode(en_str))
        return tokenized_pt, tokenized_en
