# Copyright Lee Group 2019
# Author: Ryan-Rhys Griffiths
"""
This script plots samples from a BayesOpt prior.
"""

from matplotlib import pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d.axes3d import Axes3D
import numpy as np
from scipy.optimize import minimize

from kernels import anisotropic_kernel
from mean_functions import zero_mean
from objective_functions import branin_function, branin_plot_function, noise_plot_function, min_branin_noise_function
from utils import my_nll_fn, neg_log_marg_lik, posterior_predictive


def compute_confidence_bounds(mean_vector, K):
    """
    Compute the upper and lower 95% (2 standard deviations) confidence bounds given
    the covariance matrix K of a multivariate normal distribution.

    :param mean_vector: mean vector of a multivariate normal.
    :param K: Covariance matrix of a multivariate normal.
    :return: upper and lower 95% confidence intervals.
    """

    assert len(mean_vector) == len(K[:, 0])
    std = np.sqrt(np.diag(K))
    upper = mean_vector + 2*std
    lower = mean_vector - 2*std

    return upper, lower


if __name__ == "__main__":

    # Draws from the prior

    fprior = False  # Determines whether samples from the prior are plotted or not.

    if fprior:

        xs = np.linspace(0.0, 10.0, num=100)  # array of sampling locations
        m = len(xs)  # number of sampling locations
        mean_vector = np.zeros((m,))
        K = np.zeros((m, m))  # covariance matrix

        for i in range(0, m):
            for j in range(0, m):
                K[i, j] = anisotropic_kernel(xs[i], xs[j], 2, 1)

        upper, lower = compute_confidence_bounds(mean_vector, K)
        y_1 = np.random.multivariate_normal(mean_vector, K)
        y_2 = np.random.multivariate_normal(mean_vector, K)
        plt.plot(xs, y_1, color='blue')
        plt.plot(xs, y_2, color='red')
        plt.fill_between(xs, upper, lower, color='gray', alpha=0.2)
        plt.xlabel('input, x')
        plt.ylabel('f(x)')
        plt.title('Samples from a BayesOpt Prior')
        plt.show()

    # Fit some Data

    # If 1D, generate sine wave data with Gaussian noise if. Matches the example of Martin Krasser.
    # If 2D, generate a noisy version of the Branin function.

    one_dimension = False  # Plot BayesOpt fit in one dimension
    two_dimensions = True  # Plot BayesOpt fit in two dimensions on the Branin function.
    two_dims_with_samples = False  # Plot BayesOpt fit in two dimensions on the Branin function with samples from the posterior.

    if one_dimension:

        xs = np.arange(-3, 4, 1).reshape(-1, 1)  # The locations of observed data points
        m = len(xs)  # number of training points
        mean_vector = zero_mean(xs)  # compute the mean vector at the observed data locations
        noise = 0.4  # sigma_n ^2, the noise level
        y = np.sin(xs) + noise * np.random.randn(*xs.shape)

        # Compute the BayesOpt posterior (generalisation of Rasmussen and Williams algorithm pg. 19)

        noise = 0.2
        xs_star = np.arange(-5, 5, 0.2).reshape(-1, 1).reshape(50,)  # test locations

        # We optimise the BayesOpt hyperparameters by minimizing the negative log marginal likelihood.

        res = minimize(my_nll_fn(xs, y, noise, kernel_function=anisotropic_kernel, mean_function=zero_mean),
                       [1.5, 1.5], bounds=((1e-5, None), (1e-5, None)), method='L-BFGS-B')

        l_opt, sigma_f_opt = res.x

        pred_mean, pred_var, K, L = posterior_predictive(xs, y, xs_star, noise, l_opt, sigma_f_opt, full_cov=True)

        nlml = neg_log_marg_lik(xs, y, noise, l_opt, sigma_f_opt, kernel=anisotropic_kernel, mean_func=zero_mean)

        y_1 = np.random.multivariate_normal(pred_mean.reshape(len(pred_mean)), pred_var)  # numpy mvn sampler expects mean vectors to be one-dimensional (m, ) as opposed to (m, 1)
        y_2 = np.random.multivariate_normal(pred_mean.reshape(len(pred_mean)), pred_var)
        y_3 = np.random.multivariate_normal(pred_mean.reshape(len(pred_mean)), pred_var)

        plt.plot(xs, y, '+', color='k', markersize='12', linewidth='8')
        plt.plot(xs_star, y_1, '-', color='orange')
        plt.plot(xs_star, y_2, '-', color='blue')
        plt.plot(xs_star, y_3, '-', color='purple')
        plt.plot(xs_star, pred_mean, '.', color='red')
        upper = pred_mean + 2 * np.sqrt(np.diag(pred_var).reshape(pred_mean.shape))
        lower = pred_mean - 2 * np.sqrt(np.diag(pred_var).reshape(pred_mean.shape))
        upper = upper.reshape(xs_star.shape)  # matplotlib complains about y1 values that aren't one-dimensional
        lower = lower.reshape(xs_star.shape)
        plt.fill_between(xs_star, upper, lower, color='gray', alpha=0.2)
        plt.xlabel('input, x')
        plt.ylabel('f(x)')
        plt.title('BayesOpt Posterior')
        plt.show()

    elif two_dimensions:

        x1 = np.arange(-5.0, 10.0, 5.0)
        x2 = np.arange(0.0, 15.0, 5.0)
        xs = np.array(np.meshgrid(x1, x2)).T.reshape(-1, 2)
        x1_star = np.arange(-5.0, 10.0, 0.5)
        x2_star = np.arange(0.0, 15.0, 0.5)
        xs_star = np.array(np.meshgrid(x1_star, x2_star)).T.reshape(-1, 2)  # Where 2 gives the dimensionality

        noise = 0.0

        X, Y = np.meshgrid(x1_star, x2_star)
        y = branin_plot_function(X, Y)
        #y = noise_plot_function(X, Y)  # using this for plotting at the moment.
        y = min_branin_noise_function(X, Y)
        CS = plt.contourf(X, Y, y, cmap=cm.inferno)
        CB = plt.colorbar(CS, shrink=0.8, extend='both')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title('Aleatoric Noise Function')
        plt.show()

        y = branin_function(xs[:, 0], xs[:, 1], noise)

        res = minimize(my_nll_fn(xs, y, noise, kernel_function=anisotropic_kernel, mean_function=zero_mean),
                       [1.5, 1.5, 1.5], bounds=((1e-5, None), (1e-5, None), (1e-5, None)), method='L-BFGS-B')

        l_opt = list(res.x[:-1])
        sigma_f_opt = res.x[-1]

        pred_mean, pred_var, K, L = posterior_predictive(xs, y, xs_star, noise, l_opt, sigma_f_opt, full_cov=False)

        plot_pred_mean = pred_mean.reshape(len(x1_star), len(x1_star)).T
        plot_pred_var = pred_var.reshape(len(x1_star), len(x1_star)).T
        X, Y = np.meshgrid(x1_star, x2_star)

        upper = plot_pred_mean + 2 * np.sqrt(plot_pred_var)
        lower = plot_pred_mean - 2 * np.sqrt(plot_pred_var)

        nlml = neg_log_marg_lik(xs, y, noise, l_opt, sigma_f_opt, kernel=anisotropic_kernel, mean_func=zero_mean)

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot_surface(X, Y, plot_pred_mean)
        ax.plot_surface(X, Y, upper, color='gray', alpha=0.4)
        ax.plot_surface(X, Y, lower, color='gray', alpha=0.4)
        ax.scatter(xs[:, 0], xs[:, 1], y, '+', color='red')
        plt.show()

    else:

        x1 = np.arange(-5.0, 10.0, 5.0)
        x2 = np.arange(0.0, 15.0, 5.0)
        xs = np.array(np.meshgrid(x1, x2)).T.reshape(-1, 2)
        x1_star = np.arange(-5.0, 10.0, 0.5)
        x2_star = np.arange(0.0, 15.0, 0.5)
        xs_star = np.array(np.meshgrid(x1_star, x2_star)).T.reshape(-1, 2)  # Where 2 gives the dimensionality

        noise = 0.2

        y = branin_function(xs[:, 0], xs[:, 1], noise)

        res = minimize(my_nll_fn(xs, y, noise, kernel_function=anisotropic_kernel, mean_function=zero_mean),
                       [1.5, 1.5, 1.5], bounds=((1e-5, None), (1e-5, None), (1e-5, None)), method='L-BFGS-B')

        l_opt1, l_opt2, sigma_f_opt = res.x
        l_opt = [l_opt1, l_opt2]

        pred_mean, pred_var, K, L = posterior_predictive(xs, y, xs_star, noise, l_opt, sigma_f_opt, full_cov=True)

        plot_pred_mean = pred_mean.reshape(len(x1_star), len(x1_star)).T
        plot_pred_var = np.diag(pred_var).reshape(len(x1_star), len(x1_star)).T  # Must make variance diagonal when full_cov is True.
        X, Y = np.meshgrid(x1_star, x2_star)

        upper = plot_pred_mean + 2 * np.sqrt(plot_pred_var)
        lower = plot_pred_mean - 2 * np.sqrt(plot_pred_var)

        mvn_sample1 = np.random.multivariate_normal(pred_mean.reshape(len(pred_mean)), pred_var)
        mvn_sample2 = np.random.multivariate_normal(pred_mean.reshape(len(pred_mean)), pred_var)
        plot_mvn_sample1 = mvn_sample1.reshape(len(x1_star), len(x1_star)).T
        plot_mvn_sample2 = mvn_sample2.reshape(len(x1_star), len(x1_star)).T

        nlml = neg_log_marg_lik(xs, y, noise, l_opt, sigma_f_opt, kernel=anisotropic_kernel, mean_func=zero_mean)

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot_surface(X, Y, plot_pred_mean)
        ax.plot_surface(X, Y, upper, color='gray', alpha=0.4)
        ax.plot_surface(X, Y, lower, color='gray', alpha=0.4)
        ax.plot_surface(X, Y, plot_mvn_sample1, color='red')
        ax.plot_surface(X, Y, plot_mvn_sample2, color='blue')
        ax.scatter(xs[:, 0], xs[:, 1], y, '+', color='red')
        plt.show()
