#!/usr/bin/env python3
import numpy as np
import pickle

class DeepNeuralNetwork:
    """Deep neural network performing binary or multiclass classification with multiple activations"""
    
    def __init__(self, nx, layers, activation='sig'):
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if not all(isinstance(x, int) and x > 0 for x in layers):
            raise TypeError("layers must be a list of positive integers")
        if activation not in ('sig', 'tanh'):
            raise ValueError("activation must be 'sig' or 'tanh'")
        
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        self.__activation = activation
        
        for l in range(self.__L):
            layer_size = layers[l]
            prev_size = nx if l == 0 else layers[l - 1]
            self.__weights[f"W{l+1}"] = np.random.randn(layer_size, prev_size) * np.sqrt(2 / prev_size)
            self.__weights[f"b{l+1}"] = np.zeros((layer_size, 1))
    
    @property
    def L(self):
        return self.__L
    
    @property
    def cache(self):
        return self.__cache
    
    @property
    def weights(self):
        return self.__weights
    
    @property
    def activation(self):
        return self.__activation

    def _activate(self, Z, layer_output=False):
        """Apply activation function"""
        if layer_output:  # output layer: sigmoid if 1 output, softmax otherwise
            if Z.shape[0] == 1:
                return 1 / (1 + np.exp(-Z))
            else:
                expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
                return expZ / np.sum(expZ, axis=0, keepdims=True)
        else:  # hidden layers
            if self.__activation == 'sig':
                return 1 / (1 + np.exp(-Z))
            elif self.__activation == 'tanh':
                return np.tanh(Z)

    def _derivative(self, A, dA=None):
        """Compute derivative for hidden layers"""
        if self.__activation == 'sig':
            return A * (1 - A)
        elif self.__activation == 'tanh':
            return 1 - A**2

    def forward_prop(self, X):
        self.__cache["A0"] = X
        A_prev = X
        for l in range(1, self.__L + 1):
            W = self.__weights[f"W{l}"]
            b = self.__weights[f"b{l}"]
            Z = W @ A_prev + b
            A = self._activate(Z, layer_output=(l == self.__L))
            self.__cache[f"A{l}"] = A
            A_prev = A
        return A, self.__cache

    def cost(self, Y, A):
        m = Y.shape[1]
        if A.shape[0] == 1:
            return -np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)) / m
        else:
            return -np.sum(Y * np.log(A + 1e-8)) / m

    def evaluate(self, X, Y):
        A, _ = self.forward_prop(X)
        if A.shape[0] == 1:
            predictions = np.where(A >= 0.5, 1, 0)
        else:
            predictions = np.argmax(A, axis=0)
        return predictions, self.cost(Y, A)

    def gradient_descent(self, Y, cache, alpha=0.05):
        m = Y.shape[1]
        L = self.__L
        A_final = cache[f"A{L}"]
        dZ = A_final - Y
        
        for l in reversed(range(1, L + 1)):
            A_prev = cache[f"A{l-1}"]
            W = self.__weights[f"W{l}"]
            b = self.__weights[f"b{l}"]
            
            dW = (1/m) * dZ @ A_prev.T
            db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
            
            if l > 1:
                A_prev_l = cache[f"A{l-1}"]
                dZ = (W.T @ dZ) * self._derivative(A_prev_l)
            
            self.__weights[f"W{l}"] -= alpha * dW
            self.__weights[f"b{l}"] -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose or graph:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")
        
        costs = []
        for i in range(iterations + 1):
            A, cache = self.forward_prop(X)
            self.gradient_descent(Y, cache, alpha)
            if i % step == 0 or i == 0 or i == iterations:
                cost = self.cost(Y, A)
                if verbose:
                    print(f"Cost after {i} iterations: {cost}")
                if graph:
                    costs.append((i, cost))
        if graph:
            import matplotlib.pyplot as plt
            x_vals, y_vals = zip(*costs)
            plt.plot(x_vals, y_vals, 'b-')
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.show()
        return self.evaluate(X, Y)

    def save(self, filename):
        if not filename.endswith(".pkl"):
            filename += ".pkl"
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        try:
            with open(filename, "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            return None
