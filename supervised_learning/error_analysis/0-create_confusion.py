#!/usr/bin/env python3
"""Script to create a confusion matrix"""

import numpy as np


def create_confusion_matrix(labels, logits):
    """Function to create a confusion matrix"""
    return np.matmul(labels.T, logits)
