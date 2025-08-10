#!/usr/bin/env python3
"""task8"""


def prune(df):
    """Takes a pd.DataFrame"""
    df_pruned = df.dropna(subset=["Close"])
    return df_pruned
