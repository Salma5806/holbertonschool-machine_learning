#!/usr/bin/env python3
"""task13"""


def analyze(df):
    """Takes a pd.DataFrame"""
    df_copy = df.copy()
    if "Timestamp" in df_copy.columns:
        df_copy = df_copy.drop(columns=["Timestamp"])
    stats = df_copy.describe()

    return stats
