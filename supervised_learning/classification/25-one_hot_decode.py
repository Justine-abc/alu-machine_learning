#!/usr/bin/env python3
import numpy as np

def one_hot_decode(one_hot):
    """
    Converts a one-hot matrix into numeric labels.
    one_hot: shape (classes, m)
    Returns: array of shape (m,)
    """
    try:
        return np.argmax(one_hot, axis=0)
    except Exception:
        return None
