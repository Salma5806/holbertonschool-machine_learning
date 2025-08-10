#!/usr/bin/env python3
"""
task 3
"""

import pandas as pd


def rename(df):
    """
    Takes a pd.DataFrame as input and performs the following:
    """
    df_modified = df.copy()
    df_modified = df_modified.rename(columns={"Timestamp": "Datetime"})
    df_modified["Datetime"] = pd.to_datetime(df_modified["Datetime"], unit="s")
    return df_modified[["Datetime", "Close"]]
