# Copyright Ryan-Rhys Griffiths 2021
# Author: Ryan-Rhys Griffiths
"""
This module contains homoscedastic and heteroscedastic variants of common synthetic functions used in Bayesian
optimisation experiments. Functions are negated in this module for maximisation problems as the acquisition functions
are always attempting to maximise the function.
"""

from matplotlib import pyplot as plt
import matplotlib.cm as cm
import numpy as np
from scipy.optimize import minimize


def hosaki_function(x1, x2, noise=0.0, heteroscedastic=False, f_plot=False):
    """
    Definition of the 2D Hosaki function defined here:
    https://al-roomi.org/benchmarks/unconstrained/2-dimensions/58-hosaki-s-function
    Bounds are [0, 5] in both x1 and x2.
    Global optimum: f(x) = -2.3458 for x = [4, 2].
    Mean is roughly -0.818
    Standard deviation is roughly 0.57
    Global optimum: f(x) = -2.668 for x = [4, 2]. For the standardised version

    :param x1: numpy array of points along the first dimension
    :param x2: numpy array of points along the second dimension
    :param noise: The noise level (defaults to zero)
    :param heteroscedastic: Whether to use the Branin function with heteroscedastic noise:
    :param fplot: Whether to plot the objective and noise functions.
    :return: The Hosaki(x1, x2) function evaluated at the points specified by x1 and x2 with noise level given by noise
    """

    # reshape inputs
    x1 = x1.reshape(len(x1), 1)
    x2 = x2.reshape(len(x2), 1)

    def hos_function(x1, x2):
        """Compute the Hosaki function"""

        interim = (1 - 8*x1 + 7*x1**2 - (7.0/3.0)*x1**3 + (1.0/4.0)*x1**4) * x2**2 * np.exp(-x2)

        # Standardise the function value using the mean and standard deviation of the function.

        return (interim + 0.817)/0.573

    f = hos_function(x1, x2)

    f += noise**2 * np.random.randn(*x1.shape)  # Add noise to the Hosaki  function f

    def hos_noise(x1, x2):
        """Noise function"""

        return 50 * (1 / ((x1 - 3.5) ** 2 + 2.5) * 1 / ((x2 - 2.0) ** 2 + 2.5))


    if heteroscedastic:
        assert noise == 0
        # Add heteroscedastic noise to the Hosaki function f.
        g = hos_noise(x1, x2).reshape(-1, 1)
        n = f + (g * np.random.randn(*x1.shape))

        return -n, g, -f

    if f_plot:

        x1_plot = np.arange(0.0, 5.0, 0.02)
        x2_plot = np.arange(0.0, 5.0, 0.02)

        X, Y = np.meshgrid(x1_plot, x2_plot)
        y = hos_function(X, Y)
        CS = plt.contourf(X, Y, y, cmap=cm.inferno)
        CB = plt.colorbar(CS, shrink=0.8, extend='both')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title('Hosaki Function')
        plt.show()
        plt.close()

        y2 = hos_noise(X, Y)
        CS = plt.contourf(X, Y, y2, cmap=cm.inferno)
        CB = plt.colorbar(CS, shrink=0.8, extend='both')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title('Hosaki Aleatoric Noise Function')
        plt.show()
        plt.close()

        y3 = y + y2
        CS = plt.contourf(X, Y, y3, cmap=cm.inferno)
        CB = plt.colorbar(CS, shrink=0.8, extend='both')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title('Hosaki Composite Function')
        plt.show()
        plt.close()

    return -f


def goldstein_price_function(x1, x2, noise=0.0, heteroscedastic=False, f_plot=False):
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
    :param f_plot: Whether to plot objective
    :return: The goldstein-Price(x1, x2) function evaluated at the points specified by x1 and x2 with noise level given by noise
    """

    # reshape inputs
    x1 = x1.reshape(len(x1), 1)
    x2 = x2.reshape(len(x2), 1)

    def gs_function(x1, x2):
        """
        Compute the objective.
        """

        x_bar_one = 4 * x1 - 2
        x_bar_two = 4 * x2 - 2

        return (1/2.427) * (np.log((1 + (x_bar_one + x_bar_two + 1)**2*
                             (19 - 14*x_bar_one + 3*x_bar_one**2 - 14*x_bar_two + 6*x_bar_one*x_bar_two + 3*x_bar_two**2))*
                            (30 + (2*x_bar_one - 3*x_bar_two)**2*
                             (18 - 32*x_bar_one + 12*x_bar_one**2 + 48*x_bar_two - 36*x_bar_one*x_bar_two + 27*x_bar_two**2))) - 8.693)

        # MATLAB implementation from https://www.sfu.ca/~ssurjano/Code/goldprscm.html
        # fact1a = (x1bar + x2bar + 1) ^ 2;
        # fact1b = 19 - 14 * x1bar + 3 * x1bar ^ 2 - 14 * x2bar + 6 * x1bar * x2bar + 3 * x2bar ^ 2;
        # fact1 = 1 + fact1a * fact1b;
        #
        # fact2a = (2 * x1bar - 3 * x2bar) ^ 2;
        # fact2b = 18 - 32 * x1bar + 12 * x1bar ^ 2 + 48 * x2bar - 36 * x1bar * x2bar + 27 * x2bar ^ 2;
        # fact2 = 30 + fact2a * fact2b;
        #
        # prod = fact1 * fact2;
        #
        # y = (log(prod) - 8.693) / 2.427;

    f = gs_function(x1, x2)
    f += noise**2 * np.random.randn(*x1.shape)  # Add noise to the Hosaki  function f

    def gs_noise(x1, x2):
        """
        Compute the heteroscedastic noise
        """

        return 3 * (1 / ((x1 - 0.5) ** 2 + 0.5) * 1 / ((x2 - 0.3) ** 2 + 0.5))

    if heteroscedastic:

        assert noise == 0
        # Add heteroscedastic noise to the goldstein-Price function f.

        g = gs_noise(x1, x2).reshape(-1, 1)
        n = f + (g * np.random.randn(*x1.shape))

        return -n, g, -f

    if f_plot:

        x1_plot = np.arange(0.0, 1.0, 0.01)
        x2_plot = np.arange(0.0, 1.0, 0.01)

        X, Y = np.meshgrid(x1_plot, x2_plot)
        y = -gs_function(X, Y)
        CS = plt.contourf(X, Y, y, cmap=cm.inferno)
        CB = plt.colorbar(CS, shrink=0.8, extend='both')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title('Goldstein-Price Function')
        plt.show()
        plt.close()

        y2 = gs_noise(X, Y)
        CS = plt.contourf(X, Y, y2, cmap=cm.inferno)
        CB = plt.colorbar(CS, shrink=0.8, extend='both')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title('Goldstein-Price Aleatoric Noise Function')
        plt.show()
        plt.close()

        y3 = y - y2
        CS = plt.contourf(X, Y, y3, cmap=cm.inferno)
        CB = plt.colorbar(CS, shrink=0.8, extend='both')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title('Goldstein-Price Composite Function')
        plt.show()
        plt.close()

    return -f


def branin_function(x1, x2, noise=0.0, heteroscedastic=False, f_plot=False):
    """
    Definition of the Standardised 2D Branin Function from Picheny et al.
    Bounds are [0, 1] in both x1 and x2.
    :param x1: numpy array of points along the first dimension
    :param x2: numpy array of points along the second dimension
    :param noise: The homoscedastic noise level (defaults to zero)
    :param heteroscedastic: Whether to use the Branin function with heteroscedastic noise:
    :param fplot: Whether to plot the functions
    https://link.springer.com/article/10.1007/s00158-013-0919-4 on [0, 1]^2
    :return: The Branin(x1, x2) function evaluated at the points specified by x1 and x2 with noise level given by noise
    """

    # reshape inputs
    x1 = x1.reshape(len(x1), 1)
    x2 = x2.reshape(len(x2), 1)

    def br_function(x1, x2):
        """Branin-Hoo Function"""

        x_bar_one = 15 * x1 - 5
        x_bar_two = 15 * x2

        return (1 / 51.95) * ((x_bar_two - ((5.1 * x_bar_one ** 2) / (4 * np.pi ** 2)) + (5 * x_bar_one / np.pi) - 6) ** 2 + (
                (10 - 10 / 8 * np.pi) * np.cos(x_bar_one)) - 44.81)


    f = br_function(x1, x2)
    f += noise**2 * np.random.randn(*x1.shape)  # Add noise to the Branin function f.

    def br_noise(x1, x2):
        """Noise function"""

        return 15 - (2.8 * x1 ** 2 + 4.8 * x2 ** 2)

    if heteroscedastic:
        assert noise == 0
        # Add heteroscedastic noise to the Branin function f.
        g = br_noise(x1, x2).reshape(-1, 1)
        n = f + (g * np.random.randn(*x1.shape))

        return -n, g, -f

    if f_plot:

        x1_plot = np.arange(0.0, 1.0, 0.01)
        x2_plot = np.arange(0.0, 1.0, 0.01)

        X, Y = np.meshgrid(x1_plot, x2_plot)
        y = br_function(X, Y)
        CS = plt.contourf(X, Y, y, cmap=cm.inferno)
        CB = plt.colorbar(CS, shrink=0.8, extend='both')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title('Branin Function')
        plt.show()
        plt.close()

        y2 = br_noise(X, Y)
        CS = plt.contourf(X, Y, y2, cmap=cm.inferno)
        CB = plt.colorbar(CS, shrink=0.8, extend='both')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title('Branin Aleatoric Noise Function')
        plt.show()
        plt.close()

        y3 = y + y2
        CS = plt.contourf(X, Y, y3, cmap=cm.inferno)
        CB = plt.colorbar(CS, shrink=0.8, extend='both')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title('Branin Composite Function')
        plt.show()
        plt.close()

    return -f


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


def opt_branin(x, heteroscedastic=False):
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


def opt_goldstein_price(x, heteroscedastic=False):
    """
    function for numerical optimisation of the goldstein-Price function
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
        f += 5 - (1 * x1 ** 2 + 3 * x2 ** 2)

    return f

# Calculate optima of functions using scipy optimiser


if __name__ == '__main__':

    # Verify with L-BFGS-B. Sanity check that all problems are minimisation problems also.

    res = minimize(opt_hosaki, x0=np.array([4, 2]), method='L-BFGS-B', bounds=((0, 5), (0, 5)))
    print(res.x)
    print('L-BFGS-B optimum of Hosaki function is: ', opt_hosaki(np.array(res.x)))

    res = minimize(opt_branin, x0=np.array([0.75, 0.25]), method='L-BFGS-B', bounds=((0, 1), (0, 1)))
    print(res.x)
    print('L-BFGS-B optimum of Branin function is: ', opt_branin(np.array(res.x)))

    res = minimize(opt_goldstein_price, x0=np.array([0.55, 1]), method='L-BFGS-B', bounds=((0, 1), (0, 1)))
    print(res.x)
    print('L-BFGS-B optimum of goldstein-Price function is: ', opt_goldstein_price(np.array(res.x)))

    # Plotting The objectives and noise functions

    x1 = np.array([0, 0])  # Dummy variables
    x2 = np.array([0, 0])

    goldstein_price_function(x1, x2, f_plot=True)
    hosaki_function(x1, x2, f_plot=True)
    branin_function(x1, x2, f_plot=True)
