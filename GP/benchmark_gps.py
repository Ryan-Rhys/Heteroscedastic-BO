# Copyright Lee Group 2019
# Author: Ryan-Rhys Griffiths
"""
This script benchmarks homoscedastic and heteroscedastic GPs on the benchmark datasets from Kersting et al.
"""

import numpy as np
from sklearn.model_selection import train_test_split

from datasets import williams_1996, silverman_1985, goldberg_1998, yuan_2004, lidar_1994, scallop_data
from gp_fitting import fit_hetero_gp, fit_homo_gp
from kernels import scipy_kernel
from mean_functions import zero_mean
from utils import posterior_predictive_krasser, nlpd, one_d_train_test_split, posterior_predictive


if __name__ == '__main__':

    xs, ys = scallop_data(fplot_data=False)

    if xs.shape[1] > 1:
        xs_train, xs_test, ys_train, ys_test = train_test_split(xs, ys, test_size=0.1)
    else:
        xs_train, ys_train, xs_test, ys_test = one_d_train_test_split(xs, ys, split_ratio=0.9)

    l_init = 1  # lengthscale to initialise the optimiser with
    sigma_f_init = 2  # signal amplitude to initialise the optimiser with
    noise = 1

    pred_mean, pred_var, nlml = fit_homo_gp(xs_train, ys_train, noise, xs_test, l_init, sigma_f_init, fplot=False)
    nlpd_val = nlpd(pred_mean, np.diag(pred_var), ys_test)  #  will only work if pred_mean has been evaluated at the same positions as the target ys.

    print(nlpd_val)

    l_noise_init = 1
    sigma_f_noise_init = 2
    gp2_noise = 4
    num_iters = 7
    sample_size = 100

    noise_func, gp2_noise, gp1_l_opt, gp1_sigma_f_opt, gp2_l_opt, gp2_sigma_f_opt, variance_estimator = \
        fit_hetero_gp(xs_train, ys_train, noise, xs_train, l_init, sigma_f_init, l_noise_init, sigma_f_noise_init, gp2_noise, num_iters, sample_size)

    pred_mean_het, pred_var_het, _, _ = posterior_predictive(xs_train, ys_train, xs_test, noise_func, gp1_l_opt, gp1_sigma_f_opt, mean_func=zero_mean, kernel=scipy_kernel)
    pred_mean_noise, _, _, _ = posterior_predictive(xs_train, variance_estimator, xs_test, gp2_noise, gp2_l_opt, gp2_sigma_f_opt, mean_func=zero_mean, kernel=scipy_kernel)

    pred_mean_noise = np.exp(pred_mean_noise)
    pred_mean_noise = np.sqrt(pred_mean_noise).reshape(len(pred_mean_noise))
    nlpd_val_het = nlpd(pred_mean_het, np.diag(pred_var_het) + pred_mean_noise, ys_test)

    print(nlpd_val_het)