#!/usr/bin/env python3
"""
Bayesian Optimization of a Neural Network using GPyOpt

This script optimizes a simple feedforward neural network on the MNIST dataset
using Bayesian Optimization from GPyOpt. Five hyperparameters are tuned:

- Number of units in the first hidden layer
- Number of units in the second hidden layer
- Dropout rate
- Learning rate
- L2 regularization weight

The optimization runs for a maximum of 30 iterations. Best checkpoints are
saved with filenames reflecting the hyperparameters, early stopping is used,
and convergence is plotted.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import GPyOpt
import matplotlib.pyplot as plt

# Load MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Flatten images
x_train = x_train.reshape(-1, 28*28)
x_test = x_test.reshape(-1, 28*28)

# Number of classes
num_classes = 10
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

# Define search space for hyperparameters
domain = [
    {'name': 'units1', 'type': 'discrete', 'domain': (32, 64, 128, 256)},
    {'name': 'units2', 'type': 'discrete', 'domain': (32, 64, 128, 256)},
    {'name': 'dropout', 'type': 'continuous', 'domain': (0.0, 0.5)},
    {'name': 'lr', 'type': 'continuous', 'domain': (1e-4, 1e-2)},
    {'name': 'l2', 'type': 'continuous', 'domain': (1e-5, 1e-2)},
]

def build_model(params):
    """Builds and compiles a Keras model from hyperparameters."""
    units1, units2, dropout, lr, l2 = params

    model = Sequential([
        Dense(int(units1), activation='relu', input_shape=(28*28,),
              kernel_regularizer=tf.keras.regularizers.l2(l2)),
        Dropout(dropout),
        Dense(int(units2), activation='relu',
              kernel_regularizer=tf.keras.regularizers.l2(l2)),
        Dropout(dropout),
        Dense(num_classes, activation='softmax')
    ])

    optimizer = Adam(learning_rate=lr)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def objective_function(params):
    """Objective function to minimize: negative validation accuracy."""
    params = params[0]  # GPyOpt passes 2D array
    model = build_model(params)

    # Filename for checkpoint
    units1, units2, dropout, lr, l2 = params
    checkpoint_file = f"checkpoint_u1{int(units1)}_u2{int(units2)}_d{dropout:.2f}_lr{lr:.5f}_l2{l2:.5f}.h5"

    # Early stopping and checkpoint
    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True),
        ModelCheckpoint(checkpoint_file, monitor='val_accuracy', save_best_only=True)
    ]

    history = model.fit(
        x_train, y_train,
        validation_split=0.2,
        epochs=20,
        batch_size=128,
        verbose=0,
        callbacks=callbacks
    )

    # Return negative val accuracy because GPyOpt minimizes
    val_acc = max(history.history['val_accuracy'])
    return np.array([[1 - val_acc]])

# Run Bayesian Optimization
optimizer = GPyOpt.methods.BayesianOptimization(
    f=objective_function,
    domain=domain,
    acquisition_type='EI',
    maximize=False
)

optimizer.run_optimization(max_iter=30)

# Save report
with open("bayes_opt.txt", "w") as f:
    f.write("Best hyperparameters found:\n")
    f.write(str(optimizer.x_opt) + "\n")
    f.write("Best validation accuracy achieved:\n")
    f.write(str(1 - optimizer.fx_opt) + "\n")

# Plot convergence
optimizer.plot_convergence()
plt.savefig("convergence.png")
plt.show()
