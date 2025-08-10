#!/usr/bin/env python3

import pandas as pd

def from_file(filename, delimiter):
    # Load data from file into DataFrame
    df = pd.read_csv(filename, delimiter=delimiter)
    return df
