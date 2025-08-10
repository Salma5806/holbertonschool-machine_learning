#!/usr/bin/env python3
"""
task 7
"""


def high(df):
    """Takes a pd.DataFrame"""
    df_sorted = df.sort_values(by="High", ascending=False)
    return df_sorted
