#!/usr/bin/env python3
"""task2"""
import pandas as pd


def from_file(filename, delimiter):
    """Reads a file and returns a dataframe"""
    return pd.read_csv(filename, delimiter=delimiter)
