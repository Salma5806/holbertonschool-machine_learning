#!/usr/bin/env python3
"""
task 5
"""


def slice(df):
    """
    Takes a pd.DataFrame
    """
    columns_to_extract = ["High", "Low", "Close", "Volume_(BTC)"]
    df_extracted = df[columns_to_extract]
    df_sliced = df_extracted.iloc[::60]

    return df_sliced
