#!/usr/bin/env python3
"""Script to create a Neuron"""

import numpy as np


class Neuron():
    """ Class Neuron """

    def __init__(self, nx):
        """Initialize neuron"""
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.W = np.random.randn(1, nx)
        self.b = 0
        self.A = 0
