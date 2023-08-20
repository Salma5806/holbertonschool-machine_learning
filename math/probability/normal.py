#!/usr/bin/env python3
""" defines Normal class that represents normal distribution """


class Normal:
    """class that represents normal distribution"""

    def __init__(self, data=None, mean=0., stddev=1.):
        """  Initialize normal"""
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            else:
                self.stddev = float(stddev)
                self.mean = float(mean)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")
            else:
                self.mean = sum(data) / len(data)
                sigma = 0
                for i in range(0, len(data)):
                    x = (data[i] - self.mean) ** 2
                    sigma += x
                self.stddev = (sigma / len(data)) ** (1 / 2)
                