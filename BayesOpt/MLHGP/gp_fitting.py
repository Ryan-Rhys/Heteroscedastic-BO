# Copyright Ryan-Rhys Griffiths 2019
# Author: Ryan-Rhys Griffiths
"""
This module contains Gaussian Process-fitting procedures for the homoscedastic and heteroscedastic
Gaussian Process implementations.
"""

from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import minimize

from kernels import scipy_kernel
from mean_functions import zero_mean
from gp_utils import neg_log_marg_lik_krasser, posterior_predictive, nll_fn_het


def fit_homo_gp(xs, ys, noise, xs_star, l_init, sigma_f_init, fplot=True, mean_func=zero_mean):
    """
    Fit a homoscedastic GP to data (xs, ys) and compute the negative log predictive density at new input locations
    xs_star.

    :param xs: input locations N x D
    :param ys: target labels
    :param noise: fixed noise level or noise function
    :param xs_star: test input locations
    :param l_init: lengthscale(s) to initialise the optimiser
    :param sigma_f_init: signal amplitude to initialise the optimiser
    :param fplot: bool indicating whether to plot the posterior predictive or not.
    :param mean_func: the mean function to use (can be zero or constant)
    :return: negative log marginal likelihood value and negative log predictive density.
    """

    soil_plot = False

    if soil_plot:

        mean_param = np.mean(ys)
        std_param = np.std(ys)
        ys = (ys - mean_param) / std_param  # standardise the y-values

    dimensionality = xs.shape[1]  # Extract the dimensionality of the input so that lengthscales are appropriate dimension
    hypers = [l_init]*dimensionality + [sigma_f_init] + [noise]  # we initialise each dimension with the same lengthscale value
    bounds = [(1e-2, 900)]*len(hypers)  # we initialise the bounds to be the same in each case

    # We fit GP1 to the data

    res = minimize(nll_fn_het(xs, ys, noise), hypers, bounds=bounds, method='L-BFGS-B')

    l_opt = np.array(res.x[:-2]).reshape(-1, 1)  # res.x[:-1]
    sigma_f_opt = res.x[-2]  # res.x[-1] before noise included
    noise = res.x[-1]

    pred_mean, pred_var, _, _ = posterior_predictive(xs, ys, xs_star, noise, l_opt, sigma_f_opt, mean_func=mean_func, kernel=scipy_kernel)
    nlml = neg_log_marg_lik_krasser(xs, ys, noise, l_opt, sigma_f_opt)

    f_print_diagnostics = False

    if f_print_diagnostics:

        print('lengthscale is: ' + str(l_opt))
        print('signal amplitude is: ' + str(sigma_f_opt))
        print('noise_opt is: ' + str(noise))
        print('negative log marginal likelihood is: ' + str(nlml))

    if fplot and soil_plot:
        plot_xs_star = xs_star.reshape(len(xs_star), )

        # Take the diagonal of the covariance matrix for plotting purposes
        plot_pred_var = np.diag(pred_var).reshape(-1, 1)

        # restandardisation code is for the soil example plotting specifically.

        upper = (pred_mean + 2 * np.sqrt(plot_pred_var)) * std_param + mean_param
        lower = (pred_mean - 2 * np.sqrt(plot_pred_var)) * std_param + mean_param
        upper = upper.reshape(plot_xs_star.shape)
        lower = lower.reshape(plot_xs_star.shape)
        ys = (ys * std_param) + mean_param
        pred_mean = (pred_mean * std_param) + mean_param
        plt.plot(xs, ys, '+', color='green', markersize='12', linewidth='8')
        plt.plot(plot_xs_star, pred_mean, '-', color='red')
        plt.fill_between(plot_xs_star, upper, lower, color='gray', alpha=0.2)
        plt.xlabel('Density, Dry Bulk ($g/cm^3$)', fontsize=14)
        plt.ylabel('Phosphorus Content ($mg/kg$)', fontsize=14)
        plt.xticks([0, 0.5, 1, 1.5])
        plt.yticks([0, 100, 200, 300])
        plt.tick_params(labelsize=13)
        #plt.title('Homoscedastic GP Posterior')
        plt.show()

    return pred_mean, pred_var, nlml


def fit_hetero_gp(xs, ys, aleatoric_noise, xs_star, l_init, sigma_f_init, l_noise_init, sigma_f_noise_init, gp2_noise,
                  num_iters, sample_size, mean_func):
    """
    Fit a heteroscedastic BayesOpt to data (xs, ys) and compute the negative log predictive density at new input locations
    xs_star.

    :param xs: input locations N x D
    :param ys: target labels
    :param aleatoric_noise: initial fixed noise level or noise function. will be learned by GP2
    :param xs_star: test input locations
    :param l_init: lengthscale(s) to initialise the optimiser
    :param sigma_f_init: signal amplitude to initialise the optimiser
    :param l_noise_init: lengthscale(s) to initialise the optimiser for the noise
    :param sigma_f_noise_init: signal amplitude to initialise the optimiser for the noise
    :param gp2_noise: the noise level for the second BayesOpt modelling the noise (noise of the noise)
    :param num_iters: number of iterations to run the most likely heteroscedastic BayesOpt algorithm.
    :param sample_size: the number of samples for the heteroscedastic BayesOpt algorithm.
    :param mean_func: the mean function to use, can be constant or zero mean.
    :return: The negative log marginal likelihood value and the negative log predictive density at the test input locations.
    """

    soil_plot = False

    if soil_plot:

        mean_param = np.mean(ys)
        std_param = np.std(ys)
        ys = (ys - mean_param) / std_param  # standardise the y-values

    dimensionality = xs.shape[1]  # in order to plot only in the 1D input case.
    gp1_hypers = [l_init]*dimensionality + [sigma_f_init]  # we initialise each dimension with the same lengthscale value
    gp2_hypers = [l_noise_init]*dimensionality + [sigma_f_noise_init]  # we initialise each dimensions with the same lengthscale value for gp2 as well.
    bounds = [(1, 900)]*len(gp1_hypers)  # we initialise the bounds to be the same in each case

    for i in range(0, num_iters):

        # We fit GP1 to the data

        gp1_res = minimize(nll_fn_het(xs, ys, aleatoric_noise), gp1_hypers, bounds=bounds, method='L-BFGS-B')
        #gp1_res = min_x

        # We collect the hyperparameters from the optimisation

        gp1_l_opt = np.array(gp1_res.x[:-1]).reshape(-1, 1)
        gp1_sigma_f_opt = gp1_res.x[-1]
        #gp1_noise_opt = gp1_res.x[-1]

        gp1_hypers = list(np.ndarray.flatten(gp1_l_opt)) + [gp1_sigma_f_opt]  # we initialise the optimisation at the next iteration with the optimised hypers

        # We compute the posterior predictive at the test locations

        gp1_pred_mean, gp1_pred_var, _, _ = posterior_predictive(xs, ys, xs_star, aleatoric_noise, gp1_l_opt, gp1_sigma_f_opt, mean_func=mean_func, kernel=scipy_kernel)

        f_print_diagnostics = False

        if f_print_diagnostics and i >= 1:

            nlml = neg_log_marg_lik_krasser(xs, ys, aleatoric_noise, gp1_l_opt, gp1_sigma_f_opt)
            print('negative log marginal likelihood on iteration {} is: '.format(i) + str(nlml))
            print('lengthscale on iteration {} is: '.format(i) + str(gp1_l_opt))
            print('signal amplitude on iteration {} is: '.format(i) + str(gp1_sigma_f_opt))

        # We plot the fit on the final iteration.

        f_gp1_plot_posterior = True

        if i == num_iters - 1 and dimensionality == 1:
            f_gp1_plot_posterior = False

        f_gp1_plot_posterior_2d = False

        # Switch designed for the scallop dataset

        if i == num_iters - 1 and dimensionality == 2:
            f_gp1_plot_posterior_2d = True

        if f_gp1_plot_posterior_2d:

            x1_star = np.arange(38.5, 41.0, 0.05)  # hardcoded limits for the scallop dataset.
            x2_star = np.arange(-74.0, -71.0, 0.05) # hardcoded limits for the scallop dataset.
            xs_star_plot = np.array(np.meshgrid(x1_star, x2_star)).T.reshape(-1, 2)  # Where 2 gives the dimensionality

            gp1_plot_pred_mean, gp1_plot_pred_var, _, _ = posterior_predictive(xs, ys, xs_star_plot, aleatoric_noise, gp1_l_opt, gp1_sigma_f_opt, mean_func=zero_mean, kernel=scipy_kernel, full_cov=False)

            gp1_plot_pred_mean = gp1_plot_pred_mean.reshape(len(x1_star), len(x2_star)).T
            gp1_plot_pred_var = gp1_plot_pred_var.reshape(len(x1_star), len(x2_star)).T
            X, Y = np.meshgrid(x1_star, x2_star)

            upper = gp1_plot_pred_mean + 2 * np.sqrt(gp1_plot_pred_var)
            lower = gp1_plot_pred_mean - 2 * np.sqrt(gp1_plot_pred_var)

            fig = plt.figure()
            ax = plt.axes(projection='3d')
            ax.plot_surface(X, Y, gp1_plot_pred_mean)
            #ax.plot_surface(X, Y, upper, color='gray', alpha=0.4)
            #ax.plot_surface(X, Y, lower, color='gray', alpha=0.4)
            ax.scatter(xs[:, 0], xs[:, 1], ys, '+', color='red')
            plt.show()

        if f_gp1_plot_posterior and soil_plot:
            if i == num_iters - 2:  # plot on the final iteration.
                gp1_plot_pred_var = np.diag(gp1_pred_var).reshape(-1, 1)  # Take the diagonal of the covariance matrix for plotting purposes
                gp1_plot_pred_var = gp1_plot_pred_var + np.square(aleatoric_noise)

                upper = (gp1_pred_mean + 2 * np.sqrt(gp1_plot_pred_var)) * std_param + mean_param
                lower = (gp1_pred_mean - 2 * np.sqrt(gp1_plot_pred_var)) * std_param + mean_param
                upper = upper.reshape(xs_star.shape)
                lower = lower.reshape(xs_star.shape)
                ys = (ys * std_param) + mean_param
                gp1_pred_mean = (gp1_pred_mean * std_param) + mean_param

                plt.plot(xs, ys, '+', color='green', markersize='12', linewidth='8')
                plt.plot(xs_star, gp1_pred_mean, '-', color='red')
                plt.fill_between(xs_star.reshape(len(xs_star),), upper.reshape(len(xs_star),), lower.reshape(len(xs_star),), color='gray', alpha=0.2)
                plt.xlabel('Density, Dry Bulk ($g/cm^3$)', fontsize=14)
                plt.ylabel('Phosphorus Content ($mg/kg$)', fontsize=14)
                #plt.title('Heteroscedastic GP Posterior')
                plt.xticks([0, 0.5, 1, 1.5])
                plt.yticks([0, 100, 200, 300])
                plt.tick_params(labelsize=13)
                plt.show()

        # We construct the most likely heteroscedastic BayesOpt noise estimator

        sample_matrix = np.zeros((len(ys), sample_size))

        for j in range(0, sample_size):
            sample_matrix[:, j] = np.random.multivariate_normal(gp1_pred_mean.reshape(len(gp1_pred_mean)), gp1_pred_var)

        variance_estimator = (0.5 / sample_size) * np.sum((ys - sample_matrix) ** 2, axis=1)  # Equation given in section 4 of Kersting et al. vector of noise for each data point.
        #variance_estimator = (ys - gp1_pred_mean)**2  # Matt's variance estimator
        variance_estimator = np.log(variance_estimator)

        # We fit a second BayesOpt to the auxiliary dataset z = (xs, variance_estimator)

        gp2_res = minimize(nll_fn_het(xs, variance_estimator, gp2_noise), gp2_hypers, bounds=bounds)

        # We collect the hyperparameters

        gp2_l_opt = np.array(gp2_res.x[:-1]).reshape(-1, 1)
        gp2_sigma_f_opt = gp2_res.x[-1]
        gp2_hypers = list(np.ndarray.flatten(gp2_l_opt)) + [gp2_sigma_f_opt]  # we initialise the optimisation at the next iteration with the optimised hypers

        # we reshape the variance estimator here so that it can be passed into posterior_predictive.

        variance_estimator = variance_estimator.reshape(len(variance_estimator), 1)

        gp2_pred_mean, gp2_pred_var, _, _ = posterior_predictive(xs, variance_estimator, xs_star, gp2_noise, gp2_l_opt, gp2_sigma_f_opt, mean_func=zero_mean, kernel=scipy_kernel)
        gp2_pred_mean = np.exp(gp2_pred_mean)
        aleatoric_noise = np.sqrt(gp2_pred_mean)

        f_gp2_plot_posterior = False

        if f_gp2_plot_posterior:
            gp2_plot_pred_var = np.diag(gp2_pred_var).reshape(-1, 1)  # Take the diagonal of the covariance matrix for plotting purposes
            plt.plot(xs, variance_estimator, '+', color='green', markersize='12', linewidth='8')
            plt.plot(xs_star, np.log(gp2_pred_mean), '-', color='red')
            upper = np.log(gp2_pred_mean) + 2 * np.sqrt(gp2_plot_pred_var)
            lower = np.log(gp2_pred_mean) - 2 * np.sqrt(gp2_plot_pred_var)
            upper = upper.reshape(xs_star.shape[0],)
            lower = lower.reshape(xs_star.shape[0],)
            plt.fill_between(xs_star.reshape(len(xs_star),), upper, lower, color='gray', alpha=0.2)
            plt.xlabel('input, x')
            plt.ylabel('log variance')
            plt.title('GP2 Posterior')
            plt.show()

    return aleatoric_noise, gp2_noise, gp1_l_opt, gp1_sigma_f_opt, gp2_l_opt, gp2_sigma_f_opt, variance_estimator
