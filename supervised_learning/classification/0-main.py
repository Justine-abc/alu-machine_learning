#!/usr/bin/env python3
"""0-main.py: Test 0-neuron.py"""

import numpy as np
from 0-neuron import Neuron

# Example data
np.random.seed(0)
X = np.random.randn(784, 10)  # 784 features, 10 examples

neuron = Neuron(X.shape[0])
print(neuron.W)
print(neuron.W.shape)
print(neuron.b)
print(neuron.A)
neuron.A = 10
print(neuron.A)
