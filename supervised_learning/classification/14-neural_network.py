#!/usr/bin/env python3
"""14. NeuralNetwork training"""

import numpy as np
from 13-neural_network import NeuralNetwork


class NeuralNetworkTrain(NeuralNetwork):
    """Extends NeuralNetwork with training functionality."""

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """Trains the neural network.

        Args:
            X (ndarray): Input data, shape (nx, m)
            Y (ndarray): Correct labels, shape (1, m)
            iterations (int): Number of iterations
            alpha (float): Learning rate

        Returns:
            tuple: Predictions and cost after training

        Raises:
            TypeError: If iterations not int or alpha not float
            ValueError: If iterations or alpha not positive
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        for i in range(iterations):
            A1, A2 = self.forward_prop(X)
            self.gradient_descent(X, Y, A1, A2, alpha)

        return self.evaluate(X, Y)
