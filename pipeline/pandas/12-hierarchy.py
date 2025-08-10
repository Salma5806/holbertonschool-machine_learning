#!/usr/bin/env python3
"""
task 12
"""

import pandas as pd

index = __import__('10-index').index


def hierarchy(df1, df2):
    """Takes two pd.DataFrame objects"""
    df1_indexed = index(df1)
    df2_indexed = index(df2)
    df1_filtered = df1_indexed[
        (df1_indexed.index >= 1417411980) & (df1_indexed.index <= 1417417980)
    ]
    df2_filtered = df2_indexed[
        (df2_indexed.index >= 1417411980) & (df2_indexed.index <= 1417417980)
    ]
    df_concatenated = pd.concat(
        [df2_filtered, df1_filtered], keys=["bitstamp", "coinbase"]
    )
    df_swapped = df_concatenated.swaplevel(0, 1)
    df_sorted = df_swapped.sort_index()

    return df_sorted
