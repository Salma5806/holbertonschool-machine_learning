#!/usr/bin/env python3
"""
task 9
"""


def fill(df):
    """Takes a pd.DataFrame"""
    df_filled = df.copy()
    df_filled = df_filled.drop(columns=["Weighted_Price"])
    df_filled["Close"] = df_filled["Close"].ffill()
    df_filled["High"] = df_filled["High"].fillna(df_filled["Close"])
    df_filled["Low"] = df_filled["Low"].fillna(df_filled["Close"])
    df_filled["Open"] = df_filled["Open"].fillna(df_filled["Close"])
    df_filled["Volume_(BTC)"] = df_filled["Volume_(BTC)"].fillna(0)
    df_filled["Volume_(Currency)"] = df_filled["Volume_(Currency)"].fillna(0)

    return df_filled
