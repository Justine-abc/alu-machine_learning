#!/usr/bin/env python3
"""Test 2-neuron.py"""
import numpy as np
from 2-neuron import Neuron

lib_train = np.load('../data/Binary_Train.npz')
X_3D, Y = lib_train['X'], lib_train['Y']
X = X_3D.reshape((X_3D.shape[0], -1)).T

np.random.seed(0)
neuron = Neuron(X.shape[0])
neuron._Neuron__b = 1  # manually set bias for testing
A = neuron.forward_prop(X)

# Check that returned A is the same as the neuron's A attribute
if A is neuron.A:
    print(A)
