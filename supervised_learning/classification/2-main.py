#!/usr/bin/env python3
"""Test 2-neuron.py"""
import numpy as np

Neuron = __import__('2-neuron').Neuron

np.random.seed(0)
neuron = Neuron(5)

print(neuron.W)
print(neuron.b)
print(neuron.A)

X = np.random.randn(5, 3)
A = neuron.forward_prop(X)
print(A)
