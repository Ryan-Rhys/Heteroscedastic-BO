# Copyright Lee Group 2019
# Author: Ryan-Rhys Griffiths
"""
This module implements GP mean functions.
"""

import numpy as np


def zero_mean(x):
    """
    Compute the zero mean function of a GP prior with the general form: N ~ (0, K(X, X)). Assumes that the output value
    is one-dimensional.

    :param x: a matrix of inputs of dimension N X D where N is the number of datapoints and D is the number of features
    :return: a matrix of zeros of dimension N X 1
    """

    zero_matrix = np.zeros((np.shape(x)[0], 1))
    return zero_matrix
