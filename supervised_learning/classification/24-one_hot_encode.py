#!/usr/bin/env python3
import numpy as np

def one_hot_encode(Y, classes):
    """
    Converts numeric labels into a one-hot matrix.
    Y: shape (m,)
    classes: number of classes
    Returns: one-hot array of shape (classes, m)
    """
    try:
        m = Y.shape[0]
        one_hot = np.zeros((classes, m))
        one_hot[Y, np.arange(m)] = 1
        return one_hot
    except Exception:
        return None
