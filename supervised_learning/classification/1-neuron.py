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

        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def w(self):
        """ return weight """
        return self.__W
    
    @property
    def b(self):
        """ return biais"""
        return self.__b
    
    @property
    def A(self):
        """return output"""
        return self.__A
