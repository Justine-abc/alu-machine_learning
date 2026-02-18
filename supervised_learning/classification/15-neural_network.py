#!/usr/bin/env python3
"""15. NeuralNetwork training with verbose and graph"""

import matplotlib.pyplot as plt
import numpy as np
from 13-neural_network import NeuralNetwork


class NeuralNetworkTrain(NeuralNetwork):
    """Extends NeuralNetwork with upgraded training features."""

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """Trains the neural network and optionally prints or graphs cost.

        Args:
            X (ndarray): Input data, shape (nx, m).
            Y (ndarray): Correct labels, shape (1, m).
            iterations (int): Number of iterations.
            alpha (float): Learning rate.
            verbose (bool): Print cost every 'step' iterations.
            graph (bool): Graph cost after training.
            step (int): Step interval for verbose output and graph.

        Returns:
            tuple: Predictions and cost after training.

        Raises:
            TypeError: If iterations or step not int, alpha not float.
            ValueError: If iterations or alpha not positive, step invalid.
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        if (verbose or graph):
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        costs = []
        steps = []

        for i in range(iterations + 1):
            A1, A2 = self.forward_prop(X)
            if i < iterations:
                self.gradient_descent(X, Y, A1, A2, alpha)

            if i % step == 0 or i == 0 or i == iterations:
                cost = self.cost(Y, A2)
                if verbose:
                    print(f"Cost after {i} iterations: {cost}")
                if graph:
                    costs.append(cost)
                    steps.append(i)

        if graph:
            plt.plot(steps, costs, 'b-')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()

        return self.evaluate(X, Y)
