#!/usr/bin/env python3
"""Task 0"""

import tensorflow_datasets as tfds
import transformers

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
        self.tokenizer_pt = transformers.AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
        self.tokenizer_en = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")

    def tokenize_dataset(self, data):
        """Tokenizes the dataset"""
        tokenizer_pt = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            (pt.numpy() for pt, _ in data), target_vocab_size=2**15
        )
        tokenizer_en = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            (en.numpy() for _, en in data), target_vocab_size=2**15
        )
        return tokenizer_pt, tokenizer_en
