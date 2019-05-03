# Copyright Lee Group 2019
# Author: Ryan-Rhys Griffiths
"""
This module contains utilities for Bayesian Optimisation. Code adapted from Martin Krasser's blog:
http://krasserm.github.io/2018/03/21/bayesian-optimization/#Optimization-algorithm
"""

import matplotlib.pyplot as plt
import numpy as np

from acquisition_functions import my_expected_improvement, my_propose_location, heteroscedastic_propose_location, heteroscedastic_expected_improvement
from bayes_opt_plot_utils import plot_acquisition, my_plot_approximation, my_het_plot_approximation
from gp_fitting import fit_hetero_gp
from objective_functions import linear_sin_noise, max_sin_noise_objective


def f(X, noise):
    """
    1D noise function defined by Krasser:

    :param X: input dimension
    :param noise: constant noise level
    :return: f(X) + noise
    """
    return -np.sin(3 * X) - X**2 + 0.7 * X + noise * np.random.randn(*X.shape)


if __name__ == '__main__':

    #bounds = np.array([[-1.0, 2.0]])  # bounds of the Bayesian Optimisation problem.
    bounds = np.array([[0, 3*np.pi]])  # bounds of linear sin wave noise problem.
    noise = 0.3

    #  Initial noisy data points.

    #X_init = np.array([[-0.9], [1.1]])
    #Y_init = f(X_init, noise)

    #X_init = np.array([[2.5], [6]])
    X_init = np.random.uniform(0, 3*np.pi, size=(2,1))
    X_init = np.arange(bounds[:, 0], bounds[:, 1], 0.01).reshape(-1, 1)
    Y_plot = max_sin_noise_objective(X_init, noise)
    Y_init = linear_sin_noise(X_init, noise)

    # Dense grid of points within bounds
    X = np.arange(bounds[:, 0], bounds[:, 1], 0.1).reshape(-1, 1)

    # Noise-free objective function values at X
    #Y = f(X, 0)
    Y = linear_sin_noise(X, 0)

    # Plot optimization objective with noise level
    plt.plot(X, Y, 'y--', lw=2, label='Noise-free objective')
    #plt.plot(X, f(X, noise), 'bx', lw=1, alpha=0.1, label='Noisy samples')
    plt.plot(X, linear_sin_noise(X, noise), 'bx', lw=1, alpha=0.1, label='Noisy samples')
    plt.plot(X_init, Y_init, 'kx', mew=3, label='Initial samples')
    plt.legend()

    # Initialize samples
    X_sample = X_init
    Y_sample = Y_init

    # Number of iterations
    n_iter = 20

    plt.figure(figsize=(12, n_iter * 3))
    plt.subplots_adjust(hspace=0.4)

    l_init = 0.5
    sigma_f_init = 0.5
    l_noise_init = 0.2
    sigma_f_noise_init = 0.2
    gp2_noise = 0.2
    num_iters = 10
    sample_size = 100

    # Code below is a test for the GP fit.

    # X_init = np.arange(bounds[:, 0], bounds[:, 1], 0.1).reshape(-1, 1)
    # Y_init = linear_sin_noise(X_init, noise)
    #
    # fit_hetero_gp(X_init, Y_init, noise, X_init, l_init, sigma_f_init, l_noise_init, sigma_f_noise_init, gp2_noise, num_iters,
    #               sample_size)

    obj_val_list = []
    best_so_far = -25

    for i in range(n_iter):

        print(i)

        # Obtain next sampling point from the acquisition function (expected_improvement)

        #X_next = my_propose_location(my_expected_improvement, X_sample, Y_sample, noise, l_init, sigma_f_init, bounds, n_restarts=25, min_val=1)

        X_next = heteroscedastic_propose_location(heteroscedastic_expected_improvement, X_sample, Y_sample, noise, l_init,
                                                  sigma_f_init, l_noise_init, sigma_f_noise_init, gp2_noise, num_iters,
                                                  sample_size, bounds, n_restarts=1, min_val=25)

        obj_val = max_sin_noise_objective(X_next, noise)
        print(obj_val)
        if obj_val > best_so_far:
            obj_val_list.append(obj_val[0][0])
            best_so_far = obj_val[0][0]
        else:
            obj_val_list.append(best_so_far)

        # Obtain next noisy sample from the objective function
        #Y_next = f(X_next, noise)
        Y_next = linear_sin_noise(X_next, noise)

        # Plot samples, surrogate function, noise-free objective and next sampling location
        plt.subplot(n_iter, 2, 2 * i + 1)
        #my_plot_approximation(X, Y, X_sample, Y_sample, noise, l_init, sigma_f_init, X_next=None, show_legend=False)
        my_het_plot_approximation(X, Y, X_sample, Y_sample, noise, l_init, sigma_f_init, l_noise_init,
                                  sigma_f_noise_init, gp2_noise, num_iters, sample_size)
        plt.title(f'Iteration {i+1}')

        plt.subplot(n_iter, 2, 2 * i + 2)
        #plot_acquisition(X, my_expected_improvement(X, X_sample, Y_sample, noise, l_init, sigma_f_init), X_next, show_legend=i == 0)
        plot_acquisition(X, heteroscedastic_expected_improvement(X, X_sample, Y_sample, noise, l_init, sigma_f_init,
                                                                 l_noise_init, sigma_f_noise_init, gp2_noise, num_iters,
                                                                 sample_size, hetero_ei=True), X_next, show_legend=i == 0)

        # Add sample to previous samples
        X_sample = np.vstack((X_sample, X_next))
        Y_sample = np.vstack((Y_sample, Y_next))

    plt.show()
    plt.plot(np.arange(1, n_iter + 1, 1), np.array(obj_val_list))
    plt.xlabel('Iteration Number')
    plt.ylabel('Objective Function Value')
    plt.title('Best value obtained so far')
    plt.show()
