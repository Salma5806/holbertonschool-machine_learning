#!/usr/bin/env python3
""" Early stopping should occur when the validation cost of the network has not decreased relative to the optimal validation cost 
by more than the threshold over a specific patience count """


def early_stopping(cost, opt_cost, threshold, patience, count):
    """Determines if you should stop gradient descent early."""
    if cost <= opt_cost - threshold:
        count = 0
        return False, count
    else:
        count += 1
        if count < patience:
            return False, count
        else:
            return True, count
