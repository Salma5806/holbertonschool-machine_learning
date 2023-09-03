#!/usr/bin/env python3
""" calculates the weighted moving average of a data set """

import numpy as np


def moving_average(data, beta):
    """
    Calculates the weighted moving average of a data set.

    Args:
        data: List of data to calculate the moving average of.
        beta: The weight used for the moving average.

    Returns:
        A list containing the moving averages of the data.
    """
    moving_avg = []
    prev_avg = 0.0

    for i, x in enumerate(data):
        bias_correction = 1 - (beta ** (i + 1))
        curr_avg = (beta * prev_avg) + ((1 - beta) * x / bias_correction)
        moving_avg.append(curr_avg)
        prev_avg = curr_avg

    return moving_avg
