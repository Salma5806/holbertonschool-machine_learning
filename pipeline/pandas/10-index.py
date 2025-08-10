#!/usr/bin/env python3
"""task10"""


def index(df):
    """Takes a pd.DataFrame"""
    df_indexed = df.set_index("Timestamp")

    return df_indexed
