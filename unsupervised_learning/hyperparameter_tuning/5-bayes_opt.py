#!/usr/bin/env python3
"""Task 5"""

import numpy as np
from scipy.stats import norm
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
        all points in the search space. Calculates the
        next best sample location"""
        mu_s, sigma_s = self.gp.predict(self.X_s)

        with np.errstate(divide='warn'):
            if self.minimize:
                imp = np.min(self.gp.Y) - mu_s - self.xsi
            else:
                imp = mu_s - np.max(self.gp.Y) - self.xsi

            Z = imp / sigma_s
            EI = imp * norm.cdf(Z) + sigma_s * norm.pdf(Z)
            EI[sigma_s == 0.0] = 0.0
            X_next = self.X_s[np.argmax(EI)]

        return X_next, EI

    def optimize(self, iterations=100):
        """Finds next sample point by maxing (or min-ing) the acquire
        function and evaluates the black-box function at this point and
        updates the Gaussian process with the new sample."""
        X_opt = Y_opt = None
        sampled_points = set()

        for _ in range(iterations):
            X_next, _ = self.acquisition()
            X_next_str = str(X_next)

            if X_next_str not in sampled_points:
                Y_new = self.f(X_next)
                sampled_points.add(X_next_str)

                if Y_opt is None or Y_new < Y_opt:
                    X_opt, Y_opt = X_next, Y_new

                self.gp.update(X_next, Y_new)
            else:
                break

        self.gp.X = self.gp.X[:-1]

        return X_opt, Y_opt
