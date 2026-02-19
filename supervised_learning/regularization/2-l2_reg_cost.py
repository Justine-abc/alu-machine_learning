#!/usr/bin/env python3
"""TensorFlow L2 Regularization Cost"""
import tensorflow as tf


def l2_reg_cost(cost):
    """
    Calculates total cost including L2 regularization
    """
    return cost + tf.losses.get_regularization_loss()
