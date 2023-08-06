#!/usr/bin/env python3
""" function that sum powers """


def summation_i_squared(n):
    """ n(n+1)(2n+1) / 6 """

    if type(n) is not int or n < 1:
        return None
    else:
        return int((n * (n + 1)) * (2 * n + 1) // 6)
