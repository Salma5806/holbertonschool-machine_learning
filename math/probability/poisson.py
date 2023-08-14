#!/usr/bin/env python3
""" Calculate a Poisson distribution """


class Poisson:
    """ Calculate a lambtha """

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
                k = int(k)
            if k < 0:
                return 0
            fact = 1
            for  i in range(1, k + 1):
                fact = fact * i
            pmf = (self.lambtha ** k) * (2.7182818285 ** -self.lambtha) / fact 
            return pmf

    def cdf(self, k):
        """
        Method Cumulative distribution function
        k: integer value of the data
        return: CFD
        """
        if type(k) is not int:
            k = int(k)
        if k < 0:
            return 0
        cdf = 0
        for i in range(k + 1):
            cdf = cdf + self.pmf(i)
        return cdf