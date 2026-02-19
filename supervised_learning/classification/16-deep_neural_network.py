#!/usr/bin/env python3
"""Deep Neural Network for binary classification"""

import numpy as np


class DeepNeuralNetwork:
    """Defines a deep neural network performing binary classification"""

    def __init__(self, nx, layers):
        """Class constructor"""

        # 1. Validate nx
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        # 2. Validate layers
        if (not isinstance(layers, list)) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        for layer in layers:
            if not isinstance(layer, int) or layer < 1:
                raise TypeError("layers must be a list of positive integers")

        # 3. Initialize attributes
        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        # 4. Initialize weights and biases (He initialization)
        for l in range(self.L):  # Only one loop
            if l == 0:
                prev_nodes = nx
            else:
                prev_nodes = layers[l - 1]

            self.weights["W{}".format(l + 1)] = (
                np.random.randn(layers[l], prev_nodes)
                * np.sqrt(2 / prev_nodes)
            )

            self.weights["b{}".format(l + 1)] = np.zeros((layers[l], 1))
