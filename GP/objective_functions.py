# Copyright Lee Group 2019
# Author: Ryan-Rhys Griffiths
"""
This module contains objective functions for Bayesian Optimisation.
"""

from matplotlib import pyplot as plt
import numpy as np
from utils import load_free_solv,get_y_and_std_from_x, load_p_inorg, load_quasar


def linear_sin_noise(X, noise, plot_sample, fplot=True):
    """
    1D noise function defined where noise increases linearly in the input domain. Bounds for a bimodal function could be
    [0, 3*pi]

    :param X: input dimension
    :param noise: noise level coefficient for linearly increasing noise
    :param plot_sample: Sample for plotting purposes (points in the input domain)
    :param fplot: Boolean indicating whether to plot the objective, samples and noise function
    :return: f(X) + noise(X)
    """

    linear_sin_noise = np.sin(X) + (noise * np.random.randn(*X.shape) * X)

    if fplot:
        plt.plot(X, linear_sin_noise, '+', color='green', markersize='12', linewidth='8', label='samples with Gaussian noise')
        plt.plot(plot_sample, np.sin(plot_sample), color='blue', label='mean of generative process')
        plt.plot(plot_sample, noise*plot_sample, color='red', label='noise function')
        plt.xlabel('x')
        plt.title('Heteroscedastic Sine Wave')
        plt.legend()
        plt.ylim(-3, 3)
        plt.xlim(0, 3*np.pi)
        plt.show()

    return linear_sin_noise


def max_sin_noise_objective(X, noise, fplot=True):
    """
    Objective function for maximising objective - aleatoric noise for the sin wave with linear noise. Used for
    monitoring the best value in the optimisation obtained so far.

    :param X: input to evaluate objective; can be an array of values
    :param noise: noise level coefficient
    :param fplot: Boolean indicating whether to plot the black-box objective
    :return: value of the black-box objective that penalises aleatoric noise, value of the noise at X
    """

    noise_value = noise * X  # value of the heteroscedastic noise at the point(s) X
    objective_value = np.sin(X) # value of the objective at the point(s) X
    composite_objective = objective_value - noise_value

    if fplot:
        plt.plot(X, composite_objective, color='purple', label='objective - aleatoric noise')
        plt.xlabel('x')
        plt.ylabel('objective(x)')
        plt.title('Black-box Objective')
        plt.ylim(-3, 1)
        plt.xlim(0, 3*np.pi)
        plt.show()

    composite_objective = float(composite_objective)
    noise_value = float(noise_value)

    return composite_objective, noise_value


def min_branin_noise_function(x1, x2):
    """
    Objective function for minimising objective + aleatoric noise

    :param x1: first input dimension
    :param x2: second input dimension
    :return: value of the black-box objective that penalises aleatoric noise.
    """

    return branin_plot_function(x1, x2) + noise_plot_function(x1, x2)


def branin_function(x1, x2, noise=0.0):
    """
    Definition of the 2D Branin Function.
    :param x1: numpy array of points along the first dimension
    :param x2: numpy array of points along the second dimension
    :param noise: The noise level (defaults to zero)
    :return: The Branin(x1, x2) function evaluated at the points specified by x1 and x2 with noise level given by noise
    """

    # Parameters of the function
    a = 1
    b = 5.1 / (4 * np.pi**2)
    c = 5 / np.pi
    r = 6
    s = 10
    t = 1 / (8 * np.pi)

    # reshape inputs
    x1 = x1.reshape(len(x1), 1)
    x2 = x2.reshape(len(x2), 1)

    f = a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s  # Compute the Branin(x1, x2) function

    f += noise**2 * np.random.randn(*x1.shape)  # Add noise to the Branin function f.

    return f


def branin_plot_function(x1, x2):
    """
    Function used for plotting contour plot of Branin-Hoo function.
    :param x1: numpy array of points along the first dimension
    :param x2: numpy array of points along the second dimension
    :return: The Branin(x1, x2) function evaluated at the points specified by x1 and x2
    """

    # Parameters of the function
    a = 1
    b = 5.1 / (4 * np.pi**2)
    c = 5 / np.pi
    r = 6
    s = 10
    t = 1 / (8 * np.pi)

    f = a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s  # Compute the Branin(x1, x2) function

    return f


def noise_plot_function(x1, x2):
    """
    Plot of the linear noise function.

    :param x1: first input dimension
    :param x2: second input dimension
    :return: value of the noise_function(x1, x2).
    """

    return 1.4*x1**2 + 0.3*x2


def heteroscedastic_branin(x1, x2):
    """
    Definition of a branin function with heteroscedastic noise

    :param x1: numpy array of points along the first dimension
    :param x2: numpy array of points along the second dimension
    :return: The Branin(x1, x2) function evaluated at the points specified by x1 and x2 with noise level given by a
             noise function.
    """

    def noise(x1, x2):
        """
        noise function.

        :param x1: numpy array of points along the first dimension
        :param x2: numpy array of points along the second dimension
        :return: heteroscedastic noise as a function of x1 and x2.
        """

        return 1.4*x1**2 + 0.3*x2

    # Parameters of the function

    a = 1
    b = 5.1 / (4 * np.pi**2)
    c = 5 / np.pi
    r = 6
    s = 10
    t = 1 / (8 * np.pi)

    # reshape inputs
    x1 = x1.reshape(len(x1), 1)
    x2 = x2.reshape(len(x2), 1)

    f = a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s  # Compute the Branin(x1, x2) function

    f += noise(x1, x2) * np.random.randn(*x1.shape)  # Add noise to the Branin function f.

    return f


def min_free_solv_objective(X):
    """
    Objective function for minimizing f(x) + s(x) (f is the function and s is the true noise).

    :param x: input to evaluate objective; can be an array of values
    :return: true function value and true noise value
    """

    #### THIS FUNCTION IS WRONG! IT SHOULD RETURN THE OBJECTIVE, EXPERIMENTAL MEAN - EXPERIMENTAL NOISE

    # Load the free solvation energies dataset
    X_ref, Y_ref, U_ref = load_free_solv()

    # Get the indices of the desired locations in the dataset
    n_X = X.shape[0]
    X_idx = np.array([]).astype(int)
    for i in range(n_X):
        x = X[i]
        x_idx = np.where((X_ref > x - 1e-6).all(axis=1) & (X_ref < x + 1e-6).all(axis=1))[0]
        X_idx = np.append(X_idx,x_idx)
    
    # Get the true function and noise values
    function_value = Y_ref[X_idx]
    noise_value = U_ref[X_idx]
    
    return function_value, noise_value

def max_p_inorg_objective(X):
    """
    Give composite objective (y - true noise) for the p_inorg dataset.

    :param X: locations at which to return the objective.
    :return: objective y - true noise
    """

    y, std = get_y_and_std_from_x(X,load_p_inorg)
    obj = y-std
    
    return obj[0]

def max_quasar_objective(X):
    """
    Give composite objective (y - true std) for the quasar dataset.

    :param X: locations at which to return the objective.
    :return: objective y - true nstd
    """

    y, std = get_y_and_std_from_x(X,load_quasar)
    obj = y-std
    
    return obj[0]


