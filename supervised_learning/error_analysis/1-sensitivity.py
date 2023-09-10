#!/usr/bin/env python3
"""Script to calculate the sensitivity in a
    confusion matrix
"""

import numpy as np


def sensitivity(confusion):
    """Function to calculate the sensitivity"""
    TP = np.diag(confusion)
    FN = np.sum(confusion, axis=1) - TP
    TPR = TP / (TP + FN)
    return TPR
