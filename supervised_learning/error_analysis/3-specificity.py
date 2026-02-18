#!/usr/bin/env python3
"""Calculates specificity"""
import numpy as np


def specificity(confusion):
    """
    confusion: (classes, classes)
    """
    classes = confusion.shape[0]
    specificity = np.zeros(classes)

    for i in range(classes):
        TP = confusion[i, i]
        FP = np.sum(confusion[:, i]) - TP
        FN = np.sum(confusion[i, :]) - TP
        TN = np.sum(confusion) - (TP + FP + FN)

        specificity[i] = TN / (TN + FP)

    return specificity
