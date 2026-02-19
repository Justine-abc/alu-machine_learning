#!/usr/bin/env python3
"""Forward Propagation with Dropout"""
import numpy as np


def softmax(Z):
    """Softmax activation"""
    exp = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return exp / np.sum(exp, axis=0, keepdims=True)


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Performs forward propagation with dropout
    """
    cache = {}
    cache["A0"] = X
    A = X

    for layer in range(1, L + 1):
        W = weights["W" + str(layer)]
        b = weights["b" + str(layer)]

        Z = np.matmul(W, A) + b

        if layer == L:
            A = softmax(Z)
        else:
            A = np.tanh(Z)
            D = np.random.rand(*A.shape) < keep_prob
            A = (A * D) / keep_prob
            cache["D" + str(layer)] = D.astype(int)

        cache["A" + str(layer)] = A

    return cache
