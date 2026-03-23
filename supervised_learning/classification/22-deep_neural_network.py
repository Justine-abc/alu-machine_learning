#!/usr/bin/env python3
"""22. DeepNeuralNetwork training"""

import numpy as np
from 21-deep_neural_network import DeepNeuralNetwork


class DeepNeuralNetwork(DeepNeuralNetwork):
    """Extends DeepNeuralNetwork with a basic training method."""

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """Trains the deep neural network.

        Args:
            X (ndarray): Input data (nx, m)
            Y (ndarray): Correct labels (1, m)
            iterations (int): Number of iterations
            alpha (float): Learning rate

        Returns:
            tuple: Predicted labels and cost
        """
        # Exception handling
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        for i in range(iterations):
            A, cache = self.forward_prop(X)
            self.gradient_descent(Y, cache, alpha)

        return self.evaluate(X, Y)
