#!/usr/bin/env python3
"""18. DeepNeuralNetwork forward propagation"""

import numpy as np
from 17-deep_neural_network import DeepNeuralNetwork


class DeepNeuralNetworkForward(DeepNeuralNetwork):
    """DeepNeuralNetwork with forward propagation."""

    def forward_prop(self, X):
        """Performs forward propagation in the deep neural network.

        Args:
            X (ndarray): Input data of shape (nx, m).

        Returns:
            tuple: Output of last layer and cache dictionary.
        """
        self._DeepNeuralNetwork__cache["A0"] = X
        for l in range(1, self.L + 1):
            W = self.weights["W{}".format(l)]
            b = self.weights["b{}".format(l)]
            A_prev = self.cache["A{}".format(l - 1)]
            Z = np.dot(W, A_prev) + b
            A = 1 / (1 + np.exp(-Z))  # Sigmoid activation
            self.cache["A{}".format(l)] = A
        return A, self.cache
