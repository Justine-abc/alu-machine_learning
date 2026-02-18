#!/usr/bin/env python3
"""20. DeepNeuralNetwork evaluate"""

import numpy as np
from 19-deep_neural_network import DeepNeuralNetwork


class DeepNeuralNetwork(DeepNeuralNetwork):
    """Extends DeepNeuralNetwork with evaluation."""

    def evaluate(self, X, Y):
        """Evaluates network predictions.

        Args:
            X (ndarray): Input data (nx, m)
            Y (ndarray): Correct labels (1, m)

        Returns:
            tuple: Predicted labels and cost
        """
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, cost
