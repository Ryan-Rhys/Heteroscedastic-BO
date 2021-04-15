# Copyright Ryan-Rhys Griffiths 2021
# Author: Ryan-Rhys Griffiths
"""
This module contains homoscedastic and heteroscedastic variants of common synthetic functions used in Bayesian
optimisation experiments.
"""

import numpy as np
from scipy.optimize import minimize


def hosaki_function(x1, x2, noise=0.0, heteroscedastic=False):
    """
    Definition of the 2D Hosaki function defined here:
    https://al-roomi.org/benchmarks/unconstrained/2-dimensions/58-hosaki-s-function
    Bounds are [0, 5] in both x1 and x2.
    Global optimum: f(x) = -2.3458 for x = [4, 2].
    Mean is roughly -0.818
    Standard deviation si roughly 0.57
    Global optimum: f(x) = -2.668 for x = [4, 2]. For the standardised version

    :param x1: numpy array of points along the first dimension
    :param x2: numpy array of points along the second dimension
    :param noise: The noise level (defaults to zero)
    :param heteroscedastic: Whether to use the Branin function with heteroscedastic noise:
    :return: The Hosaki(x1, x2) function evaluated at the points specified by x1 and x2 with noise level given by noise
    """

    # reshape inputs
    x1 = x1.reshape(len(x1), 1)
    x2 = x2.reshape(len(x2), 1)

    f = (1 - 8*x1 + 7*x1**2 - (7.0/3.0)*x1**3 + (1.0/4.0)*x1**4) * x2**2 * np.exp(-x2)

    # Standardise the function value using the mean and standard deviation of the function.

    f = (f + 0.817)/0.573

    f += noise**2 * np.random.randn(*x1.shape)  # Add noise to the Hosaki  function f

    if heteroscedastic:
        # Add heteroscedastic noise to the Hosaki function f.
        f += (15 - (2.8 * x1 ** 2 + 4.8 * x2 ** 2)).reshape(-1, 1) * np.random.randn(*x1.shape)

    return -f


def goldman_price_function(x1, x2, noise=0.0, heteroscedastic=False):
    """
    Definition of the 2D Hosaki function defined here:
    https://www.sfu.ca/~ssurjano/goldpr.html and defined as in Picheny et al. 2012
    Bounds are [0, 1] in both x1 and x2.
    Global optimum: f(x) =
    Mean is zero
    Standard deviation is 1

    :param x1: numpy array of points along the first dimension
    :param x2: numpy array of points along the second dimension
    :param noise: The noise level (defaults to zero)
    :param heteroscedastic: Whether to use the Branin function with heteroscedastic noise:
    :return: The Goldman-Price(x1, x2) function evaluated at the points specified by x1 and x2 with noise level given by noise
    """

    # reshape inputs
    x1 = x1.reshape(len(x1), 1)
    x2 = x2.reshape(len(x2), 1)

    x_bar_one = 4*x1 - 2
    x_bar_two = 4*x2 - 2

    f = (1/2.427) * (np.log((1 + (x_bar_one + x_bar_two + 1)**2*
                             (19 - 14*x_bar_one + 3*x_bar_one**2 - 14*x_bar_two + 6*x_bar_one*x_bar_two + 3*x_bar_two**2))*
                            (30 + (2*x_bar_one - 3*x_bar_two)**2*
                             (18 - 32*x_bar_one + 12*x_bar_one**2 + 48*x_bar_two - 36*x_bar_one*x_bar_two + 27*x_bar_two**2))) - 8.693)

    f += noise**2 * np.random.randn(*x1.shape)  # Add noise to the Hosaki  function f

    if heteroscedastic:
        # Add heteroscedastic noise to the Goldman-Price function f.
        f += (15 - (2.8 * x1 ** 2 + 4.8 * x2 ** 2)).reshape(-1, 1) * np.random.randn(*x1.shape)

    return -f


def branin_func(x1, x2, noise=0.0, heteroscedastic=False):
    """
    Definition of the 2D Branin Function.
    :param x1: numpy array of points along the first dimension
    :param x2: numpy array of points along the second dimension
    :param noise: The homoscedastic noise level (defaults to zero)
    :param heteroscedastic: Whether to use the Branin function with heteroscedastic noise:
    https://link.springer.com/article/10.1007/s00158-013-0919-4 on [0, 1]^2
    :return: The Branin(x1, x2) function evaluated at the points specified by x1 and x2 with noise level given by noise
    """

    # reshape inputs
    x1 = x1.reshape(len(x1), 1)
    x2 = x2.reshape(len(x2), 1)

    x_bar_one = 15*x1 - 5
    x_bar_two = 15*x2
    f = (1/51.95)*((x_bar_two - ((5.1*x_bar_one**2)/(4*np.pi**2)) + (5*x_bar_one/np.pi) - 6)**2 +
                   ((10 - 10/8*np.pi)*np.cos(x_bar_one)) - 44.81)

    f += noise**2 * np.random.randn(*x1.shape)  # Add noise to the Branin function f.

    if heteroscedastic:
        # Add heteroscedastic noise to the Branin function f.
        f += (15 - (2.8 * x1 ** 2 + 4.8 * x2 ** 2)).reshape(-1, 1) * np.random.randn(*x1.shape)

    return f


def opt_hosaki(x, heteroscedastic=False):
    """
    function for numerical optimisation of the Hosaki function
    """

    x1 = x[0]
    x2 = x[1]

    f = (1 - 8*x1 + 7*x1**2 - (7.0/3.0)*x1**3 + (1.0/4.0)*x1**4) * x2**2 * np.exp(-x2)

    # Standardise the function value using the mean and standard deviation of the function.

    f = (f + 0.817)/0.573

    # Add noise function as penalty to the minimisation problem
    if heteroscedastic:
        f += 15 - (2.8 * x1 ** 2 + 4.8 * x2 ** 2)

    return f


def opt_branin_func(x, heteroscedastic=False):
    """
    function for numerical optimisation of the Branin function
    """

    x1 = x[0]
    x2 = x[1]

    x_bar_one = 15*x1 - 5
    x_bar_two = 15*x2
    f = (1/51.95)*((x_bar_two - ((5.1*x_bar_one**2)/(4*np.pi**2)) + (5*x_bar_one/np.pi) - 6)**2 +
                   ((10 - 10/8*np.pi)*np.cos(x_bar_one)) - 44.81)

    # Add noise function as penalty to the minimisation problem
    if heteroscedastic:
        f += 15 - (2.8 * x1 ** 2 + 4.8 * x2 ** 2)

    return f


def opt_goldman_price(x, heteroscedastic=False):
    """
    function for numerical optimisation of the Goldman-Price function
    """

    x1 = x[0]
    x2 = x[1]

    x_bar_one = 4*x1 - 2
    x_bar_two = 4*x2 - 2

    f = (1/2.427) * (np.log((1 + (x_bar_one + x_bar_two + 1)**2 *
                             (19 - 14*x_bar_one + 3*x_bar_one**2 - 14*x_bar_two + 6*x_bar_one*x_bar_two + 3*x_bar_two**2))*
                            (30 + (2*x_bar_one - 3*x_bar_two)**2 *
                             (18 - 32*x_bar_one + 12*x_bar_one**2 + 48*x_bar_two - 36*x_bar_one*x_bar_two + 27*x_bar_two**2))) - 8.693)

    # Add noise function as penalty to the minimisation problem
    if heteroscedastic:
        f += 15 - (2.8 * x1 ** 2 + 4.8 * x2 ** 2)

    return f

# Calculate optima of functions using scipy optimiser


if __name__ == '__main__':

    # Verify with L-BFGS-B

    res = minimize(opt_hosaki, x0=np.array([4, 2]), method='L-BFGS-B', bounds=((0, 5), (0, 5)))
    print(res.x)
    print('L-BFGS-B optimum of Hosaki function is: ', opt_hosaki(np.array(res.x)))

    res = minimize(opt_branin_func, x0=np.array([0.75, 0.25]), method='L-BFGS-B', bounds=((0, 1), (0, 1)))
    print(res.x)
    print('L-BFGS-B optimum of Branin function is: ', opt_branin_func(np.array(res.x)))

    res = minimize(opt_goldman_price, x0=np.array([0.55, 1]), method='L-BFGS-B', bounds=((0, 1), (0, 1)))
    print(res.x)
    print('L-BFGS-B optimum of Goldman-Price function is: ', opt_goldman_price(np.array(res.x)))
