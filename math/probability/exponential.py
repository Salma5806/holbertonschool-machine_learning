#!/usr/bin/env python3
"""represents an exponential distribution"""


class Exponential:
    """ represents an exponential dist """
    def __init__(self, data=None, lambtha=1.):
        """ Initialize exponential """
        if data is None:
            if lambtha < 1:
                raise ValueError("lambtha must be a positive value")
            else:
                self.lambtha = float(lambtha)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")
            else:
                lambtha = float(len(data) / sum(data))
                self.lambtha = lambtha

    def pdf(self, x):
        """ calculates the value of the PDF """
        if x < 0:
            return 0
        else:
            pdf = self.lambtha * (2.7182818285 ** (-self.lambtha * x))
            return pdf

