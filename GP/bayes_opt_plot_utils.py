# Author: Martin Krasser
"""
This module contains objective functions for Bayesian Optimisation. Code from:
https://github.com/krasserm/bayesian-machine-learning/blob/master/bayesian_optimization_util.py
"""


import numpy as np
import matplotlib.pyplot as plt

from bo_gp_fit_predict import bo_predict_hetero_gp, bo_fit_hetero_gp
from gp_fitting import fit_homo_gp


def plot_approximation(gpr, X, Y, X_sample, Y_sample, X_next=None, show_legend=False):
    mu, std = gpr.predict(X, return_std=True)
    plt.fill_between(X.ravel(),
                     mu.ravel() + 1.96 * std,
                     mu.ravel() - 1.96 * std,
                     alpha=0.1)
    plt.plot(X, Y, 'y--', lw=1, label='Noise-free objective')
    plt.plot(X, mu, 'b-', lw=1, label='Surrogate function')
    plt.plot(X_sample, Y_sample, 'kx', mew=3, label='Noisy samples')
    if X_next:
        plt.axvline(x=X_next, ls='--', c='k', lw=1)
    if show_legend:
        plt.legend()


def my_plot_approximation(X, Y, X_sample, Y_sample, noise, l_init, sigma_f_init, X_next=None, show_legend=False):
    """
    Adjustment of the above function that uses my GP implementation as opposed to the sklearn one.
    """
    mu, var, _ = fit_homo_gp(X_sample, Y_sample, noise, X, l_init, sigma_f_init, fplot=False)
    std = np.sqrt(np.diag(var))
    plt.fill_between(X.ravel(),
                     mu.ravel() + 1.96 * std,
                     mu.ravel() - 1.96 * std,
                     alpha=0.1)
    plt.plot(X, Y, 'y--', lw=1, label='Noise-free objective')
    plt.plot(X, mu, 'b-', lw=1, label='Surrogate function')
    plt.plot(X_sample, Y_sample, 'kx', mew=3, label='Noisy samples')
    if X_next:
        plt.axvline(x=X_next, ls='--', c='k', lw=1)
    if show_legend:
        plt.legend()


def my_het_plot_approximation(X, Y, X_sample, Y_sample, noise, l_init, sigma_f_init, l_noise_init,
                              sigma_f_noise_init, gp2_noise, num_iters, sample_size, X_next=None, show_legend=False):
    """
    Adjustment of the above function that uses my GP implementation as opposed to the sklearn one.
    """
    noise, gp2_noise, gp1_l_opt, gp1_sigma_f_opt, gp2_l_opt, gp2_sigma_f_opt, variance_estimator = \
        bo_fit_hetero_gp(X_sample, Y_sample, noise, l_init, sigma_f_init, l_noise_init, sigma_f_noise_init, gp2_noise, num_iters, sample_size)

    mu, var, _ = bo_predict_hetero_gp(X_sample, Y_sample, variance_estimator, X, noise, gp1_l_opt, gp1_sigma_f_opt,
                                      gp2_noise, gp2_l_opt, gp2_sigma_f_opt)  # returns the variance as a vector
    std = np.sqrt(var)
    plt.fill_between(X.ravel(),
                     mu.ravel() + 1.96 * std,
                     mu.ravel() - 1.96 * std,
                     alpha=0.1)
    plt.plot(X, Y, 'y--', lw=1, label='Noise-free objective')
    plt.plot(X, mu, 'b-', lw=1, label='Surrogate function')
    plt.plot(X_sample, Y_sample, 'kx', mew=3, label='Noisy samples')
    if X_next:
        plt.axvline(x=X_next, ls='--', c='k', lw=1)
    if show_legend:
        plt.legend()


def plot_acquisition(X, Y, X_next, show_legend=False):
    plt.plot(X, Y, 'r-', lw=1, label='Acquisition function')
    plt.axvline(x=X_next, ls='--', c='k', lw=1, label='Next sampling location')
    if show_legend:
        plt.legend()


def plot_convergence(X_sample, Y_sample, n_init=2):
    plt.figure(figsize=(12, 3))

    x = X_sample[n_init:].ravel()
    y = Y_sample[n_init:].ravel()
    r = range(1, len(x) + 1)

    x_neighbor_dist = [np.abs(a - b) for a, b in zip(x, x[1:])]
    y_max_watermark = np.maximum.accumulate(y)

    plt.subplot(1, 2, 1)
    plt.plot(r[1:], x_neighbor_dist, 'bo-')
    plt.xlabel('Iteration')
    plt.ylabel('Distance')
    plt.title('Distance between consecutive x\'s')

    plt.subplot(1, 2, 2)
    plt.plot(r, y_max_watermark, 'ro-')
    plt.xlabel('Iteration')
    plt.ylabel('Best Y')
    plt.title('Value of best selected sample')
