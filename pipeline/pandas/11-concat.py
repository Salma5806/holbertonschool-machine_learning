#!/usr/bin/env python3
"""task 11"""
import pandas as pd


def concat(df1, df2):
    """Takes two pd.DataFrame objects"""
    index = __import__('10-index').index
    df1_indexed = index(df1)
    df2_indexed = index(df2)
    df2_filtered = df2_indexed[df2_indexed.index <= 1417411920]
    df_concatenated = pd.concat(
        [df2_filtered, df1_indexed], keys=["bitstamp", "coinbase"]
    )

    return df_concatenated
