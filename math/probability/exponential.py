#!/usr/bin/env python3
"""represents an exponential distribution"""


def __init__(self, data=None, lambtha=1.):
    """ Initialize exponential """
    if data is None:
        if lambtha < 0 : 
            return ValueError("lambtha must be a positive value")
        else:
            self.lambtha = float(lambtha)  
    else:
        if type(data) is not list:
            return TypeError("data must be a list")
        if len(data) < 2:
            return ValueError("data must contain multiple values")
        else:
            self.lambtha = sum(data) / len(data)   