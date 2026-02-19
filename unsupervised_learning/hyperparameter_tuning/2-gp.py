#!/usr/bin/env python3
"""Gaussian Process with update"""

import numpy as np


class GaussianProcess:
    """Noiseless 1D Gaussian Process"""

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(self.X, self.X)

    def kernel(self, X1, X2):
        sqdist = np.sum(X1**2, 1).reshape(-1, 1) + \
                 np.sum(X2**2, 1) - 2 * np.matmul(X1, X2.T)

        return self.sigma_f**2 * np.exp(-0.5 / self.l**2 * sqdist)

    def predict(self, X_s):
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s)
        K_inv = np.linalg.inv(self.K)

        mu = K_s.T @ K_inv @ self.Y
        mu = mu.reshape(-1)

        cov = K_ss - K_s.T @ K_inv @ K_s
        sigma = np.diag(cov)

        return mu, sigma

    def update(self, X_new, Y_new):
        """Update GP with new sample"""
        self.X = np.vstack((self.X, X_new.reshape(-1, 1)))
        self.Y = np.vstack((self.Y, Y_new.reshape(-1, 1)))
        self.K = self.kernel(self.X, self.X)
