#!/usr/bin/env python3
"""19. DeepNeuralNetwork cost"""

import numpy as np
from 18-deep_neural_network import DeepNeuralNetworkForward


class DeepNeuralNetwork(DeepNeuralNetworkForward):
    """Extends DeepNeuralNetwork with cost computation."""

    def cost(self, Y, A):
        """Calculates logistic regression cost.

        Args:
            Y (ndarray): True labels of shape (1, m)
            A (ndarray): Activated output of shape (1, m)

        Returns:
            float: Cost
        """
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)) / m
        return cost
