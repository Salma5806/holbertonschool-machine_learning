#!/usr/bin/env python3
"""Program that created a pd.DataFrame from a file"""
import pandas as pd


def from_file(filename, delimiter):
    """Reads a file and returns a dataframe"""
    return pd.read_csv(filename, delimiter=delimiter)
