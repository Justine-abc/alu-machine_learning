#!/usr/bin/env python3
"""Bayesian Optimization: Optimization Loop"""

import numpy as np
BO_base = __import__('4-bayes_opt').BayesianOptimization


class BayesianOptimization(BO_base):
    """Extends BayesianOptimization with an optimize() method"""

    def optimize(self, iterations=100):
        """
        Optimizes the black-box function using Expected Improvement.

        iterations: max number of iterations

        Returns:
        X_opt: numpy.ndarray of shape (1,)
        Y_opt: numpy.ndarray of shape (1,)
        """
        for _ in range(iterations):
            # Get next best sample from acquisition
            X_next, _ = self.acquisition()

            # Stop if X_next has already been sampled
            if np.any(np.isclose(X_next, self.gp.X, atol=1e-8)):
                break

            # Sample the function at X_next
            Y_next = self.f(X_next)

            # Update the Gaussian process with the new point
            self.gp.update(X_next, Y_next)

        # Determine optimal point and value
        if self.minimize:
            idx = np.argmin(self.gp.Y)
        else:
            idx = np.argmax(self.gp.Y)

        X_opt = self.gp.X[idx].reshape(1,)
        Y_opt = self.gp.Y[idx].reshape(1,)
        return X_opt, Y_opt
