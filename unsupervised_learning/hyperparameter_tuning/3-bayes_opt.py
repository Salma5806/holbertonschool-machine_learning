#!/usr/bin/env python3
"""Task 3"""

import numpy as np
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """Bayesian Optimization Class"""
    def __init__(self, f, X_init, Y_init, bounds, ac_samples,
                 l=1, sigma_f=1, xsi=0.01, minimize=True):
        """Initializes Bayesian Optimization"""
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        self.X_s = np.linspace(bounds[0], bounds[1], ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """Calculates acquisition function values for
        all points in the search space"""
        mu_s, sigma_s = self.gp.predict(self.X_s)
        if self.minimize:
            return mu_s - self.xsi * sigma_s
        else:
            return mu_s + self.xsi * sigma_s

    def optimize(self):
        """Finds next sample point by maxing (or min-ing) the acquire
        function and evaluates the black-box function at this point and
        updates the Gaussian process with the new sample."""
        next_sample = self.X_s[np.argmax(self.acquisition())]
        next_output = self.f(next_sample)
        self.gp.update(next_sample, next_output)
