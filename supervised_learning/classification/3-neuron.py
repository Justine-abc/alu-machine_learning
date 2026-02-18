#!/usr/bin/env python3
"""Neuron class with cost calculation"""
import numpy as np
from 2-neuron import Neuron


class Neuron(Neuron):
    """Neuron with cost calculation"""

    def cost(self, Y, A):
        """Calculates the logistic regression cost"""
        m = Y.shape[1]
        cost = - (1 / m) * np.sum(
            Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        )
        return cost
