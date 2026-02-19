#!/usr/bin/env python3
"""Deep Neural Network for binary classification"""

import numpy as np


class DeepNeuralNetwork:
    """Defines a deep neural network performing binary classification"""

    def __init__(self, nx, layers):

        # Validate nx
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        # Validate layers
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        if not all(isinstance(x, int) and x > 0 for x in layers):
            raise TypeError("layers must be a list of positive integers")

        # Initialize attributes
        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        # ONE AND ONLY ONE LOOP
        for l in range(self.L):
            prev = nx if l == 0 else layers[l - 1]

            self.weights["W{}".format(l + 1)] = (
                np.random.randn(layers[l], prev) * np.sqrt(2 / prev)
            )

            self.weights["b{}".format(l + 1)] = np.zeros((layers[l], 1))
