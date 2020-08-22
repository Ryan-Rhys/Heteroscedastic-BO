# Copyright Ryan-Rhys Griffiths 2019
"""
This module contains code from Martin Krasser's blog:
http://krasserm.github.io/2018/03/21/bayesian-optimization/#Optimization-algorithm
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

from acquisition_functions import expected_improvement
from bayes_opt_plot_utils import plot_acquisition, plot_approximation


def f(X, noise):
    """
    1D noise function defined by Krasser:

    :param X: input dimension
    :param noise: constant noise level
    :return: f(X) + noise
    """
    return -np.sin(3 * X) - X**2 + 0.7 * X + noise * np.random.randn(*X.shape)


def propose_location(acquisition, X_sample, Y_sample, gpr, bounds, n_restarts=25):

    """
    Proposes the next sampling point by optimizing the acquisition function.
    :param: acquisition: Acquisition function.
    :param: X_sample: Sample locations (n x d).
    :param: Y_sample: Sample values (n x 1).
    :param: gpr: A GaussianProcessRegressor fitted to samples.
    :return: Location of the acquisition function maximum.
    """

    dim = X_sample.shape[1]
    min_val = 1
    min_x = None

    def min_obj(X):
        # Minimization objective is the negative acquisition function
        return -acquisition(X.reshape(-1, dim), X_sample, gpr)

    # Find the best optimum by starting from n_restart different random points.
    for x0 in np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_restarts, dim)):
        res = minimize(min_obj, x0=x0, bounds=bounds, method='L-BFGS-B')
        if res.fun < min_val:
            min_val = res.fun[0]
            min_x = res.x

    return min_x.reshape(-1, 1)


if __name__ == '__main__':

    import time
    start = time.time()

    bounds = np.array([[-1.0, 2.0]])  # bounds of the Bayesian Optimisation problem.
    noise = 0.2

    #  Initial noisy data points.

    X_init = np.array([[-0.9], [1.1]])
    Y_init = f(X_init, noise)

    # Dense grid of points within bounds
    X = np.arange(bounds[:, 0], bounds[:, 1], 0.01).reshape(-1, 1)

    # Noise-free objective function values at X
    Y = f(X, 0)

    # Plot optimization objective with noise level
    plt.plot(X, Y, 'y--', lw=2, label='Noise-free objective')
    plt.plot(X, f(X, noise), 'bx', lw=1, alpha=0.1, label='Noisy samples')
    plt.plot(X_init, Y_init, 'kx', mew=3, label='Initial samples')
    plt.legend()

    # Gaussian process with Matern kernel as surrogate model
    m52 = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
    gpr = GaussianProcessRegressor(kernel=m52, alpha=noise ** 2)

    # Initialize samples
    X_sample = X_init
    Y_sample = Y_init

    # Number of iterations
    n_iter = 10

    plt.figure(figsize=(12, n_iter * 3))
    plt.subplots_adjust(hspace=0.4)

    for i in range(n_iter):

        # Update Gaussian process with existing samples
        gpr.fit(X_sample, Y_sample)

        # Obtain next sampling point from the acquisition function (expected_improvement)
        X_next = propose_location(expected_improvement, X_sample, Y_sample, gpr, bounds)

        # Obtain next noisy sample from the objective function
        Y_next = f(X_next, noise)

        # Plot samples, surrogate function, noise-free objective and next sampling location
        plt.subplot(n_iter, 2, 2 * i + 1)
        plot_approximation(gpr, X, Y, X_sample, Y_sample, X_next, show_legend=i == 0)
        plt.title(f'Iteration {i+1}')

        plt.subplot(n_iter, 2, 2 * i + 2)
        plot_acquisition(X, expected_improvement(X, X_sample, gpr), X_next, show_legend=i == 0)

        # Add sample to previous samples
        X_sample = np.vstack((X_sample, X_next))
        Y_sample = np.vstack((Y_sample, Y_next))

    plt.show()
