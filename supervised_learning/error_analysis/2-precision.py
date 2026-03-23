#!/usr/bin/env python3
"""Calculates precision"""
import numpy as np


def precision(confusion):
    """
    confusion: (classes, classes)
    """
    TP = np.diag(confusion)
    FP = np.sum(confusion, axis=0) - TP
    return TP / (TP + FP)
