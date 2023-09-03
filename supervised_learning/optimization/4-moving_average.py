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
    weighted_sum = 0.0
    beta_power = 1.0

    for i, value in enumerate(data):
        weighted_sum = (beta * weighted_sum) + ((1 - beta) * value)
        bias_correction = weighted_sum / (1 - beta_power)
        moving_averages.append(bias_correction)
        beta_power *= beta

    return moving_averages