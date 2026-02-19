#!/usr/bin/env python3
"""Gradient Descent with L2 Regularization"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Updates weights using gradient descent with L2 regularization
    """
    m = Y.shape[1]
    dZ = cache["A" + str(L)] - Y

    for layer in reversed(range(1, L + 1)):
        A_prev = cache["A" + str(layer - 1)]
        W = weights["W" + str(layer)]

        dW = (1 / m) * np.matmul(dZ, A_prev.T) + \
             (lambtha / m) * W
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        if layer > 1:
            A_prev_layer = cache["A" + str(layer - 1)]
            dZ = np.matmul(W.T, dZ) * (1 - np.power(A_prev_layer, 2))

        weights["W" + str(layer)] -= alpha * dW
        weights["b" + str(layer)] -= alpha * db
