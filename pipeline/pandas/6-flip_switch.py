#!/usr/bin/env python3
"""
task 6
"""


def flip_switch(df):
    """Takes a pd.DataFrame"""
    df_sorted = df.sort_index(ascending=False)
    df_transposed = df_sorted.T

    return df_transposed
