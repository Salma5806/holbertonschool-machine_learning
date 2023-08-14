#!/usr/bin/env python3
""" Calculate a Poisson distribution """
import numpy as np
import math

class Poisson:
    """ Calculate a lambtha """
    e = 2.72

    def __init__(self, data=None, lambtha=1.):
        """ Initialize poisson """
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            else:
                self.lambtha = float(lambtha)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            else:
                self.lambtha = sum(data) / len(data)
    def pmf(self, k):
        """ calculates the value of the PMF """
        if k: 
            if type(k) is not int:
                self.k = int(k)
            if k < 0:
                return 0
        return ((self.lambtha ** k) * (2.7182818285 ** -(self.lambtha))) / np.math.factorial(k)        
