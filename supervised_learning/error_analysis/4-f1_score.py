#!/usr/bin/env python3
"""Calculates F1 score"""
import numpy as np

sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """
    confusion: (classes, classes)
    """
    recall = sensitivity(confusion)
    prec = precision(confusion)

    return 2 * (prec * recall) / (prec + recall)
