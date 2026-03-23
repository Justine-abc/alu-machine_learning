#!/usr/bin/env python3
"""Creates a confusion matrix"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """
    labels: one-hot (m, classes)
    logits: one-hot (m, classes)
    """
    true = np.argmax(labels, axis=1)
    pred = np.argmax(logits, axis=1)

    classes = labels.shape[1]
    confusion = np.zeros((classes, classes))

    for i in range(len(true)):
        confusion[true[i], pred[i]] += 1

    return confusion
