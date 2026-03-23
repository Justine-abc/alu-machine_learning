#!/usr/bin/env python3
"""Gradient Descent with Dropout"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache,
                             alpha, keep_prob, L):
    """
    Updates the weights of a neural network
    using gradient descent with Dropout regularization
    """
    m = Y.shape[1]
    dZ = cache["A" + str(L)] - Y

    for layer in reversed(range(1, L + 1)):
        A_prev = cache["A" + str(layer - 1)]
        W_current = weights["W" + str(layer)]

        dW = (1 / m) * np.matmul(dZ, A_prev.T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        if layer > 1:
            dA_prev = np.matmul(W_current.T, dZ)
            D_prev = cache["D" + str(layer - 1)]
            dA_prev = (dA_prev * D_prev) / keep_prob
            A_prev = cache["A" + str(layer - 1)]
            dZ = dA_prev * (1 - np.power(A_prev, 2))

        weights["W" + str(layer)] -= alpha * dW
        weights["b" + str(layer)] -= alpha * db
