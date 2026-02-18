#!/usr/bin/env python3
"""Neuron class with forward propagation"""
import numpy as np
from 1-neuron import Neuron


class Neuron(Neuron):
    """Neuron with forward propagation"""

    def forward_prop(self, X):
        """Calculates forward propagation using sigmoid"""
        z = np.dot(self.W, X) + self.b
        self._Neuron__A = 1 / (1 + np.exp(-z))
        return self.A
