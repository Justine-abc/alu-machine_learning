#!/usr/bin/env python3
"""Bayesian Optimization: Expected Improvement Acquisition"""

import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """Performs Bayesian optimization on a noiseless 1D Gaussian process"""

    def __init__(self, f, X_init, Y_init, bounds, ac_samples,
                 l=1, sigma_f=1, xsi=0.01, minimize=True):
        """
        f: black-box function
        X_init: numpy.ndarray of shape (t, 1)
        Y_init: numpy.ndarray of shape (t, 1)
        bounds: tuple (min, max)
        ac_samples: number of acquisition samples
        l: length parameter
        sigma_f: std for GP
        xsi: exploration-exploitation factor
        minimize: True for minimization
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l=l, sigma_f=sigma_f)
        self.X_s = np.linspace(bounds[0], bounds[1], ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """
        Calculates the next best sample location using
        the Expected Improvement acquisition function.

        Returns:
        X_next: numpy.ndarray of shape (1,)
        EI: numpy.ndarray of shape (ac_samples,)
        """
        mu, sigma = self.gp.predict(self.X_s)
        if self.minimize:
            best = np.min(self.gp.Y)
            imp = best - mu - self.xsi
        else:
            best = np.max(self.gp.Y)
            imp = mu - best - self.xsi

        with np.errstate(divide='ignore', invalid='ignore'):
            Z = imp / sigma
            EI = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            EI[sigma == 0.0] = 0.0

        X_next = self.X_s[np.argmax(EI)]
        return X_next.reshape(1,), EI
