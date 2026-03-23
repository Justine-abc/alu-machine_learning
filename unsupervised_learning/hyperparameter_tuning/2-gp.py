#!/usr/bin/env python3
"""Gaussian Process with update"""

import numpy as np


class GaussianProcess:
    """Represents a noiseless 1D Gaussian Process"""

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
        Initializes the Gaussian Process.

        X_init: numpy.ndarray of shape (t, 1)
        Y_init: numpy.ndarray of shape (t, 1)
        l: length parameter
        sigma_f: standard deviation parameter
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(self.X, self.X)

    def kernel(self, X1, X2):
        """
        Calculates the covariance kernel matrix
        using the Radial Basis Function (RBF).

        X1: numpy.ndarray of shape (m, 1)
        X2: numpy.ndarray of shape (n, 1)

        Returns:
        numpy.ndarray of shape (m, n)
        """
        x1_sq = np.sum(X1 ** 2, axis=1).reshape(-1, 1)
        x2_sq = np.sum(X2 ** 2, axis=1)
        cross = 2 * np.matmul(X1, X2.T)

        sqdist = x1_sq + x2_sq - cross

        return self.sigma_f ** 2 * np.exp(
            -0.5 * sqdist / (self.l ** 2)
        )

    def predict(self, X_s):
        """
        Predicts the mean and variance of points
        in the Gaussian Process.

        X_s: numpy.ndarray of shape (s, 1)

        Returns:
        mu: numpy.ndarray of shape (s,)
        sigma: numpy.ndarray of shape (s,)
        """
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s)
        K_inv = np.linalg.inv(self.K)

        mu = K_s.T @ K_inv @ self.Y
        mu = mu.reshape(-1)

        cov = K_ss - K_s.T @ K_inv @ K_s
        sigma = np.diag(cov)

        return mu, sigma

    def update(self, X_new, Y_new):
        """
        Updates the Gaussian Process with a new sample.

        X_new: numpy.ndarray of shape (1,)
        Y_new: numpy.ndarray of shape (1,)
        """
        self.X = np.vstack((self.X, X_new.reshape(-1, 1)))
        self.Y = np.vstack((self.Y, Y_new.reshape(-1, 1)))
        self.K = self.kernel(self.X, self.X)
