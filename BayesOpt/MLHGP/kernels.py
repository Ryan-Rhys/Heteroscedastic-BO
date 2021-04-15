# Copyright Ryan-Rhys Griffiths 2019
# Author: Ryan-Rhys Griffiths
"""
This module contains definitions of common Gaussian Process kernels.
"""

import numpy as np
from scipy.spatial.distance import cdist


def kernel(X1, X2, l, sigma_f):
    """
    Implementation of squared exponential kernel from http://krasserm.github.io/2018/03/19/gaussian-processes/
    Isotropic squared exponential kernel. Computes a covariance matrix from points in X1 and X2.

    :param X1: Array of m points (m x 1)
    :param X2: Array of n points (n x 1)
    :param l: horizontal lengthscale
    :param sigma_f: vertical lengthscale
    :return: Covariance matrix (m x n)
    """
    sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
    return sigma_f**2 * np.exp(-0.5 / l**2 * sqdist)


def scipy_kernel(X1, X2, l, sigma_f):
    """
    Scipy implementation of the squared exponential kernel which deals with arbitrary dimensionality. Martin Krasser's
    implementation above only deals with the one-dimensional case. For safety this kernel should probably only be used
    for the multidimensional case.

    :param X1: Array of m points (m x d)
    :param X2: Array of n points (n x d)
    :param l: horizontal lengthscale(s)
    :param sigma_f: vertical lengthscale
    :return: Covariance matrix (m x n)
    """
    l = np.array(l).reshape(-1, 1)
    num_dims = X1.shape[1]
    l_matrix = np.zeros((num_dims, num_dims))
    np.fill_diagonal(l_matrix, 1/l)  # lengthscale gets squared later on
    reduced_X1 = X1@l_matrix  # we right multiply by a diagonal matrix with 1/lengthscale on the diagonals
    reduced_X2 = X2@l_matrix
    pairwise_sq_dists = cdist(reduced_X1, reduced_X2, 'sqeuclidean')
    pairwise_sq_dists = pairwise_sq_dists.clip(1e-30)  # clamping works very well cf. GPyTorch code.

    kernel_choice = 'matern_1/2'  # One of ['matern_1/2', 'matern_3/2', 'matern_5/2', 'rbf']

    if kernel_choice == 'matern_1/2':
        K = np.exp(-np.sqrt(pairwise_sq_dists))
    elif kernel_choice == 'matern_3/2':
        prefactor = 1 + (np.sqrt(3) * np.sqrt(pairwise_sq_dists))
        K = np.exp(-np.sqrt(3)*np.sqrt(pairwise_sq_dists))*prefactor
    elif kernel_choice == 'matern_5/2':
        prefactor = 1 + (np.sqrt(5) * np.sqrt(pairwise_sq_dists)) + (5/3 * pairwise_sq_dists)
        K = np.exp(-np.sqrt(5)*np.sqrt(pairwise_sq_dists))*prefactor
    elif kernel_choice == 'rbf':
        K = np.exp(-0.5 * pairwise_sq_dists)
    else:
        raise(RuntimeError('Invalid kernel choice specified. Options are [\'matern_1/2\', \'matern_3/2\', '
                           '\'matern_5/2\', \'rbf\']'))

    return sigma_f**2 * K


def tanimoto_kernel(X1, X2, sigma_f):
    """
    implementation of the tanimoto kernel. Unused currently but may be used for molecule experiments.

    :param X1: Array of m points (m x d)
    :param X2: Array of n points (n x d)
    :param sigma_f: vertical lengthscale
    :return Covariance matrix (m x n)
    """

    X1s = np.sum(np.square(X1), axis=-1)  # Squared L2-norm of X
    X2s = np.sum(np.square(X2), axis=-1)  # Squared L2-norm of X
    outer_product = np.tensordot(X1, X2, axes=([-1], [-1]))  # outer product of the matrices X1 and X2
    denominator = -outer_product + (X1s[:, np.newaxis] + X2s)
    K = sigma_f**2 * outer_product/denominator

    return K


def anisotropic_kernel(X1, X2, l, sigma_f):
    """
    Implementation of anisotropic squared exponential kernel. Computes a covariance matrix from points in X1 and X2.
    Contains a separate lengthscale for each of d dimensions.

    :param X1: Array of m points (m x d)
    :param X2: Array of n points (n x d)
    :param l: List of lengthscales of length d
    :param sigma_f: vertical lengthscale
    :return: Covariance matrix (m x n)
    """

    if isinstance(l, int) or isinstance(l, float):
        l = np.array(l).reshape(1, 1)  # Ensure that l is a numpy array
    else:
        l = np.array(l).reshape(len(l), 1)

    # reshape rank 1 arrays so that they have a second index

    X1 = np.atleast_1d(X1)
    X2 = np.atleast_1d(X2)

    if len(X1.shape) == 1:
        X1 = X1.reshape(-1, 1)
    if len(X2.shape) == 1:
        X2 = X2.reshape(-1, 1)

    num_dims = X1.shape[1]
    assert num_dims == len(l)  # There must be a lengthscale per dimension

    l_matrix = np.zeros((num_dims, num_dims))
    np.fill_diagonal(l_matrix, 1/l)  # lengthscale gets squared later on
    reduced_X1 = X1@l_matrix  # we right multiply by a diagonal matrix with 1/lengthscale on the diagonals
    reduced_X2 = X2@l_matrix
    sqdist = np.sum(reduced_X1**2, 1).reshape(-1, 1) + np.sum(reduced_X2**2, 1) - 2 * np.dot(reduced_X1, reduced_X2.T)
    sqdist = sqdist.clip(1e-30)

    kernel_choice = 'matern_1/2'  # One of ['matern_1/2', 'matern_3/2', 'matern_5/2', 'rbf']

    if kernel_choice == 'matern_1/2':
        K = np.exp(-np.sqrt(sqdist))
    elif kernel_choice == 'matern_3/2':
        prefactor = 1 + (np.sqrt(3) * np.sqrt(sqdist))
        K = np.exp(-np.sqrt(3)*np.sqrt(sqdist))*prefactor
    elif kernel_choice == 'matern_5/2':
        prefactor = 1 + (np.sqrt(5) * np.sqrt(sqdist)) + (5/3 * sqdist)
        K = np.exp(-np.sqrt(5)*np.sqrt(sqdist))*prefactor
    elif kernel_choice == 'rbf':
        K = np.exp(-0.5 * sqdist)
    else:
        raise(RuntimeError('Invalid kernel choice specified. Options are [\'matern_1/2\', \'matern_3/2\', '
                           '\'matern_5/2\', \'rbf\']'))

    return sigma_f**2 * K


# Functions below used only for testing purposes.

def sq_exp(x_1, x_2, lengthscale, sigma):
    """
    Compute the squared exponential kernel function between the points x_1 and x_2 in 1D

    :param x_1: numpy array specifying the first input location
    :param x_2: numpy array specifying the second input location
    :param sigma: the amplitude hyperparameter (vertical lengthscale); a float
    :param lengthscale: float specifying the lengthscale for the BayesOpt
    :return: k(x_1, x_2) where k is the kernel function.
    """

    assert lengthscale != 0  # Causes zero division error.

    k = sigma**2 * np.exp(-np.square(x_1 - x_2)/(2*lengthscale**2))
    return k


def compute_kernel_matrix_sq_exp(X, lengthscale, sigma):
    """
    Compute the kernel matrix given an N X D input matrix and a kernel function.

    :param X: matrix of inputs of input N X D
    :param sigma: signal amplitude hyperparameter
    :param lengthscale: lengthscale hyperparameter
    :return: kernel (covariance) matrix of X using the squared exponential kernel.
    """

    assert lengthscale != 0  # Causes zero division error.

    N = X.shape[0]
    K = np.zeros((N, N))
    i = 0
    for input_one in X:
        j = 0
        for input_two in X:
            K[i, j] = sigma**2 * np.exp((-np.linalg.norm(input_one - input_two)**2)/(2*lengthscale**2))
            j += 1
        i += 1
    return K
