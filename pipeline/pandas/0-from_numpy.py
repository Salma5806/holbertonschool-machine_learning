#!/usr/bin/env python3
"""
task 0
"""

import pandas as pd


def from_numpy(array):
    """Creates a pd.DataFrame from a np.ndarray"""
    num_cols = array.shape[1] if len(array.shape) > 1 else 1
    columns = [chr(ord('A') + i) for i in range(num_cols)]
    return pd.DataFrame(array, columns=columns)
