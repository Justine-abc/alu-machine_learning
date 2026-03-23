#!/usr/bin/env python3
"""Create Layer with Dropout"""
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """
    Creates a dense layer followed by dropout
    """
    initializer = tf.contrib.layers.variance_scaling_initializer(
        mode="FAN_AVG")

    layer = tf.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=initializer
    )(prev)

    return tf.layers.dropout(layer, rate=1 - keep_prob)
