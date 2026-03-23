#!/usr/bin/env python3
"""17. Privatized DeepNeuralNetwork class"""

import numpy as np


class DeepNeuralNetwork:
    """Defines a deep neural network performing binary classification."""

    def __init__(self, nx, layers):
        """Initialize the deep neural network with private attributes.

        Args:
            nx (int): Number of input features.
            layers (list): Number of nodes in each layer.

        Raises:
            TypeError: If nx is not int, or layers is not list of positive ints.
            ValueError: If nx < 1.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if not all(isinstance(n, int) and n > 0 for n in layers):
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for l in range(self.__L):
            layer_size = layers[l]
            prev_size = nx if l == 0 else layers[l - 1]
            self.__weights["W{}".format(l + 1)] = (np.random.randn(layer_size, prev_size) *
                                                   np.sqrt(2 / prev_size))
            self.__weights["b{}".format(l + 1)] = np.zeros((layer_size, 1))

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights
