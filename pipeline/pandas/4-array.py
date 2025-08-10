#!/usr/bin/env python3
"""
Module that contains a function to convert DataFrame columns to numpy array
"""


def array(df):
    """
    Takes a pd.DataFrame as input and performs the following:
    """
    last_10_rows = df[["High", "Close"]].tail(10)
    return last_10_rows.to_numpy()
