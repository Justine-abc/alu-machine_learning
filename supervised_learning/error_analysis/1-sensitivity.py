#!/usr/bin/env python3
"""Calculates sensitivity"""
import numpy as np


def sensitivity(confusion):
    """
    confusion: (classes, classes)
    """
    TP = np.diag(confusion)
    FN = np.sum(confusion, axis=1) - TP
    return TP / (TP + FN)
