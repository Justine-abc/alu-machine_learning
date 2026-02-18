#!/usr/bin/env python3
"""21. DeepNeuralNetwork gradient descent"""

import numpy as np
from 20-deep_neural_network import DeepNeuralNetwork


class DeepNeuralNetwork(DeepNeuralNetwork):
    """Extends DeepNeuralNetwork with gradient descent."""

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Performs one pass of gradient descent on the network.

        Args:
            Y (ndarray): Correct labels (1, m)
            cache (dict): Dictionary of all intermediary values
            alpha (float): Learning rate
        """
        m = Y.shape[1]
        L = self.L
        weights_copy = self.weights.copy()
        dZ_prev = 0

        for l in reversed(range(1, L + 1)):
            A = cache["A{}".format(l)]
            A_prev = cache["A{}".format(l - 1)]
            W = weights_copy["W{}".format(l)]

            if l == L:
                dZ = A - Y
            else:
                dZ = np.dot(weights_copy["W{}".format(l + 1)].T, dZ_prev) * (A * (1 - A))

            dW = np.dot(dZ, A_prev.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m

            self.weights["W{}".format(l)] -= alpha * dW
            self.weights["b{}".format(l)] -= alpha * db

            dZ_prev = dZ
