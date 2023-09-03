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
    moving_averages = []
    weighted_sum = 0
    beta_power = 1

    for i, value in range(len(data)):
        weighted_sum = (beta * weighted_sum) + ((1 - beta) * data[i])
        bias_correction = 1 - beta ** (i + 1)
        new_weight = weighted_sum / bias_correction
        moving_averages.append(new_weight)

    return moving_averages