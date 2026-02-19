#!/usr/bin/env python3
"""Gradient Descent with Dropout"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache,
                             alpha, keep_prob, L):
    """
    Updates weights using gradient descent with dropout
    """
    m = Y.shape[1]
    dZ = cache["A" + str(L)] - Y

    for layer in reversed(range(1, L + 1)):
        A_prev = cache["A" + str(layer - 1)]
        W = weights["W" + str(layer)]

        dW = (1 / m) * np.matmul(dZ, A_prev.T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        weights["W" + str(layer)] -= alpha * dW
        weights["b" + str(layer)] -= alpha * db

        if layer > 1:
            dA_prev = np.matmul(W.T, dZ)
            D = cache["D" + str(layer - 1)]
            dA_prev = (dA_prev * D) / keep_prob
            A_prev = cache["A" + str(layer - 1)]
            dZ = dA_prev * (1 - np.power(A_prev, 2))
