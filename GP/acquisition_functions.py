# Copyright Lee Group 2019
# Author: Ryan-Rhys Griffiths
"""
This module contains acquisition functions for Bayesian Optimisation.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

from bo_gp_fit_predict import bo_fit_homo_gp, bo_predict_homo_gp, bo_fit_hetero_gp, bo_predict_hetero_gp


def expected_improvement(X, X_sample, gpr, xi=0.01):
    """
    Computes the EI at points X based on existing samples X_sample and Y_sample using a Gaussian process surrogate model.
    :param X: Points at which EI should be computed (n x d).
    :param X_sample: Sample locations (m x d).
    :param gpr: A GaussianProcessRegressor fitted to samples.
    :param xi: Exploitation-exploration trade-off parameter.
    :return: Expected improvements at points X.
    """

    mu, sigma = gpr.predict(X, return_std=True)  # sigma is the standard deviation and not the variance.
    mu_sample = gpr.predict(X_sample)

    sigma = sigma.reshape(-1, X_sample.shape[1])

    # Needed for noise-based model,
    # otherwise use np.max(Y_sample).
    # See also section 2.4 in [...]
    mu_sample_opt = np.max(mu_sample)

    with np.errstate(divide='warn'):
        imp = mu - mu_sample_opt - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0

    return ei


def my_expected_improvement(X, X_sample, Y_sample, noise, l_opt, sigma_f_opt, mu_sample_opt):
    """
    Computes the EI using a homoscedastic GP.

    :param X: Test locations (n x d)
    :param X_sample: Sample locations (m x d).
    :param Y_sample:  Noise-corrupted values at the sample locations (m x 1).
    :param noise: noise level in the latent function.
    :param l_opt: optimised lengthscale(s) of the GP kernel.
    :param sigma_f_opt: optimised vertical lengthscale of the GP kernel.
    :param mu_sample_opt: incumbent eta
    :return: Expected improvements at points X.
    """

    mu, var = bo_predict_homo_gp(X_sample, Y_sample, X, noise, l_opt, sigma_f_opt)
    std = np.sqrt(np.diag(var))

    with np.errstate(divide='warn'):
        imp = mu - mu_sample_opt
        Z = imp / std
        ei = imp * norm.cdf(Z) + std * norm.pdf(Z)
        ei[std == 0.0] = 0.0

    return ei


def augmented_expected_improvement(X, X_sample, Y_sample, noise, l_opt, sigma_f_opt, mu_sample_opt):
    """
    Computes the AEI using a homoscedastic GP.

    :param X: Test locations (n x d)
    :param X_sample: Sample locations (m x d).
    :param Y_sample:  Noise-corrupted values at the sample locations (m x 1).
    :param noise: noise level in the latent function.
    :param l_opt: optimised lengthscale(s) of the GP kernel.
    :param sigma_f_opt: optimised vertical lengthscale of the GP kernel.
    :param mu_sample_opt: incumbent eta
    :return: Expected improvements at points X.
    """

    mu, var = bo_predict_homo_gp(X_sample, Y_sample, X, noise, l_opt, sigma_f_opt)
    std = np.sqrt(np.diag(var))

    with np.errstate(divide='warn'):
        imp = mu - mu_sample_opt
        Z = imp / std
        ei = imp * norm.cdf(Z) + std * norm.pdf(Z)
        ei[std == 0.0] = 0.0
        aei = ei*(1 - noise/np.sqrt(noise**2 + var**2))

    return aei


def heteroscedastic_expected_improvement(X, X_sample, Y_sample, variance_estimator, noise_func, gp1_l_opt,
                                         gp1_sigma_f_opt, gp2_l_opt, gp2_sigma_f_opt, gp2_noise, mu_sample_opt,
                                         hetero_ei=True):
    """
    Computes the EI using a heteroscedastic GP.

    :param X: Test locations (n x d)
    :param X_sample: Sample locations (m x d)
    :param Y_sample: Sample labels (m x 1)
    :param variance_estimator: estimated variance at sample locations (m x 1)
    :param noise_func: learned noise function
    :param gp1_l_opt: optimised lengthscale for GP1
    :param gp1_sigma_f_opt: optimised lengthscale for GP1
    :param gp2_l_opt: optimised lengthscale for GP2
    :param gp2_sigma_f_opt: optimised signal amplitude for GP2
    :param gp2_noise: GP2 noise level
    :param mu_sample_opt: incumbent eta
    :param hetero_ei: whether to use the ei minus one standard deviation as acquisition function
    :return: expected improvement at the test locations.
    """

    mu, var, aleatoric_std = bo_predict_hetero_gp(X_sample, Y_sample, variance_estimator, X, noise_func, gp1_l_opt, gp1_sigma_f_opt, gp2_noise, gp2_l_opt, gp2_sigma_f_opt)
    std = np.sqrt(np.diag(var))

    if hetero_ei:

        with np.errstate(divide='warn'):
            imp = mu - mu_sample_opt
            Z = imp / std
            ei = imp * norm.cdf(Z) + std * norm.pdf(Z)
            ei -= aleatoric_std
            ei[std == 0.0] = 0.0
    else:

        with np.errstate(divide='warn'):
            imp = mu - mu_sample_opt
            Z = imp / std
            ei = imp * norm.cdf(Z) + std * norm.pdf(Z)
            ei[std == 0.0] = 0.0

    return ei


def heteroscedastic_augmented_expected_improvement(X, X_sample, Y_sample, variance_estimator, noise_func, gp1_l_opt,
                                                   gp1_sigma_f_opt, gp2_l_opt, gp2_sigma_f_opt, gp2_noise, mu_sample_opt,
                                                   hetero_ei=True):
    """
    Computes the AEI using a heteroscedastic GP.

    :param X: Test locations (n x d)
    :param X_sample: Sample locations (m x d)
    :param Y_sample: Sample labels (m x 1)
    :param variance_estimator: estimated variance at sample locations (m x 1)
    :param noise_func: learned noise function
    :param gp1_l_opt: optimised lengthscale for GP1
    :param gp1_sigma_f_opt: optimised lengthscale for GP1
    :param gp2_l_opt: optimised lengthscale for GP2
    :param gp2_sigma_f_opt: optimised signal amplitude for GP2
    :param gp2_noise: GP2 noise level
    :param mu_sample_opt: incumbent eta
    :param hetero_ei: whether to use the ei minus one standard deviation as acquisition function
    :return: expected improvement at the test locations.
    """

    # var is epistemic + aleatoric uncertainty

    mu, var, aleatoric_std = bo_predict_hetero_gp(X_sample, Y_sample, variance_estimator, X, noise_func, gp1_l_opt, gp1_sigma_f_opt, gp2_noise, gp2_l_opt, gp2_sigma_f_opt)
    epistemic_unc = var - aleatoric_std
    #std = np.sqrt(np.diag(var))
    std = np.sqrt(np.diag(epistemic_unc))


    if hetero_ei:

        with np.errstate(divide='warn'):
            imp = mu - mu_sample_opt
            Z = imp / std
            ei = imp * norm.cdf(Z) + std * norm.pdf(Z)
            ei[std == 0.0] = 0.0
            aei = ei*(1 - aleatoric_std/np.sqrt(aleatoric_std**2 + epistemic_unc**2))

        return aei

    else:

        with np.errstate(divide='warn'):
            imp = mu - mu_sample_opt
            Z = imp / std
            ei = imp * norm.cdf(Z) + std * norm.pdf(Z)
            ei[std == 0.0] = 0.0

        return ei


def my_propose_location(acquisition, X_sample, Y_sample, noise, l_init, sigma_f_init, bounds, plot_sample, n_restarts=1,
                        min_val=1):
    """
    Proposes the next sampling point by optimising the acquisition function.

    :param acquisition: Acquisition function.
    :param X_sample: Sample locations (n x d).
    :param Y_sample: Noise-corrupted sample values (n x 1).
    :param noise: noise level.
    :param l_init: GP lengthscale to start optimisation with.
    :param sigma_f_init: vertical lengthscale to start optimisation with.
    :param bounds: bounds of the BO problem.
    :param plot_sample: for plotting the predictive mean and variance. Same for as X_sample but usually bigger.
    :param n_restarts: number of restarts for the optimiser.
    :param min_val: minimum value to do better than (will likely change depending on the problem).
    :return: Location of the acquisition function maximum.
    """

    dim = X_sample.shape[1]
    min_x = None

    l_opt, sigma_f_opt, noise = bo_fit_homo_gp(X_sample, Y_sample, noise, l_init, sigma_f_init)
    #l_opt, sigma_f_opt = bo_fit_homo_gp(X_sample, Y_sample, noise, l_init, sigma_f_init)

    # Purpose of this line is just to plot.

    _, _ = bo_predict_homo_gp(X_sample, Y_sample, plot_sample, noise, l_opt, sigma_f_opt, f_plot=False)  # predictive mean at test locations (uniformly spaced in the bounds.

    mu_sample, _ = bo_predict_homo_gp(X_sample, Y_sample, X_sample, noise, l_opt, sigma_f_opt)  # predictive mean for sample locations
    mu_sample_opt = np.max(mu_sample)

    def min_obj(X):
        """
        Minimisation objective is the negative acquisition function.

        :param X: points at which objective is evaluated.
        :return: minimisation objective.
        """

        X = X.reshape(-1, 1).T  # Might have to be changed for higher dimensions (have added .T since writing this)

        return -acquisition(X, X_sample, Y_sample, noise, l_opt, sigma_f_opt, mu_sample_opt)

    # Find the best optimum by starting from n_restart different random points.

    if dim == 1:  # change bounds for a single dimensions. Added for UCI dataset NAS experiments
        for x0 in np.random.uniform(bounds[0], bounds[1], size=(n_restarts, dim)):
            res = minimize(min_obj, x0=x0, bounds=bounds.reshape(-1, 2), method='L-BFGS-B')
            if res.fun < min_val:
                min_val = res.fun[0]
                min_x = res.x
    else:
        for x0 in np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_restarts, dim)):
            res = minimize(min_obj, x0=x0, bounds=bounds, method='L-BFGS-B')
            if res.fun < min_val:
                min_val = res.fun[0]
                min_x = res.x

    return min_x.reshape(-1, 1).T


def heteroscedastic_propose_location(acquisition, X_sample, Y_sample, noise, l_init, sigma_f_init,
                                     l_noise_init, sigma_f_noise_init, gp2_noise, num_iters, sample_size,
                                     bounds, plot_sample, n_restarts=25, min_val=1):
    """
    Proposes the next sampling point by optimising the acquisition function.

    :param acquisition: Heteroscedastic Acquisition function.
    :param X_sample: Sample locations (m x d).
    :param Y_sample: Sample values (m x 1).
    :param noise: noise function.
    :param l_init: GP lengthscale to start optimisation with.
    :param sigma_f_init: vertical lengthscale to start optimisation with.
    :param l_noise_init: GP2 lengthscale
    :param sigma_f_noise_init: GP2 vertical lengthscale
    :param gp2_noise: gp2 noise levels
    :param num_iters: number of iterations to run the heteroscedastic GP
    :param sample_size: samples for variance estimator.
    :param bounds: bounds of the BO problem.
    :param plot_sample: for plotting the predictive mean and variance. Same for as X_sample but usually bigger.
    :param n_restarts: number of restarts for the optimiser.
    :param min_val: minimum value to do better than (will likely change depending on the problem).
    :return: Location of the acquisition function maximum.
    """

    dim = X_sample.shape[1]
    min_x = None

    # Set f_plot to true if you want to test whether the first iteration of the heteroscedastic GP is equivalent to the homoscedastic GP.

    noise_func, gp2_noise, gp1_l_opt, gp1_sigma_f_opt, gp2_l_opt, gp2_sigma_f_opt, variance_estimator = \
        bo_fit_hetero_gp(X_sample, Y_sample, noise, l_init, sigma_f_init, l_noise_init, sigma_f_noise_init, gp2_noise, num_iters, sample_size, plot_sample, f_plot=False)

    # Purpose of this line is for plotting.

    _, _, _ = bo_predict_hetero_gp(X_sample, Y_sample, variance_estimator, X_sample, noise_func, gp1_l_opt, gp1_sigma_f_opt, gp2_noise, gp2_l_opt, gp2_sigma_f_opt, f_plot=False, f_plot2=False)

    mu_sample, _, _ = bo_predict_hetero_gp(X_sample, Y_sample, variance_estimator, X_sample, noise_func, gp1_l_opt, gp1_sigma_f_opt, gp2_noise, gp2_l_opt, gp2_sigma_f_opt)
    mu_sample_opt = np.max(mu_sample)

    def min_obj(X):
        """
        Minimisation objective is the negative acquisition function.

        :param X: points at which objective is evaluated.
        :return: minimisation objective.
        """

        X = X.reshape(-1, 1).T  # Might have to be changed for higher dimensions.

        #return -acquisition(X, X_sample, Y_sample, noise, l_init, sigma_f_init, l_noise_init, sigma_f_noise_init, gp2_noise, num_iters, sample_size)
        return -acquisition(X, X_sample, Y_sample, variance_estimator, noise_func, gp1_l_opt, gp1_sigma_f_opt, gp2_l_opt, gp2_sigma_f_opt, gp2_noise, mu_sample_opt, hetero_ei=True)

    if dim == 1:  # change bounds for a single dimensions. Added for UCI dataset NAS experiments
        for x0 in np.random.uniform(bounds[0], bounds[1], size=(n_restarts, dim)):
            res = minimize(min_obj, x0=x0, bounds=bounds.reshape(-1, 2), method='L-BFGS-B')
            if res.fun < min_val:
                min_val = res.fun[0]
                min_x = res.x
    else:
        # Find the best optimum by starting from n_restart different random points.
        for x0 in np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_restarts, dim)):
            res = minimize(min_obj, x0=x0, bounds=bounds, method='L-BFGS-B')
            if res.fun < min_val:
                min_val = res.fun[0]
                min_x = res.x

    return min_x.reshape(-1, 1).T  # added the tranpose for (2,) cases. shouldn't affect (1,) cases.
