#!/usr/bin/env python3
"""23. DeepNeuralNetwork training with verbose and graphing"""

import numpy as np
import matplotlib.pyplot as plt
from 22-deep_neural_network import DeepNeuralNetwork


class DeepNeuralNetwork(DeepNeuralNetwork):
    """Extends DeepNeuralNetwork with enhanced training."""

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """Trains the deep neural network with options to print and graph cost.

        Args:
            X (ndarray): Input data (nx, m)
            Y (ndarray): Correct labels (1, m)
            iterations (int): Number of iterations
            alpha (float): Learning rate
            verbose (bool): Print cost during training
            graph (bool): Plot cost curve after training
            step (int): Steps to print/graph

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
        if (verbose or graph):
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        costs = []

        for i in range(iterations + 1):
            A, cache = self.forward_prop(X)
            if i < iterations:
                self.gradient_descent(Y, cache, alpha)

            if i % step == 0 or i == iterations:
                cost = self.cost(Y, A)
                costs.append((i, cost))
                if verbose:
                    print(f"Cost after {i} iterations: {cost}")

        if graph:
            iters, cs = zip(*costs)
            plt.plot(iters, cs, 'b-')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()

        return self.evaluate(X, Y)
