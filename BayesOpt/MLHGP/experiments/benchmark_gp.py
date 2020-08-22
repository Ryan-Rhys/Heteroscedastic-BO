# Copyright Ryan-Rhys Griffiths 2019
# Author: Ryan-Rhys Griffiths
"""
This script benchmarks homoscedastic and heteroscedastic GPs on the benchmark datasets from Kersting et al. in terms of
their negative log predictive density (NLPD) on heldout test points.
"""

import argparse

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from dataset_loaders import DatasetLoader
from gp_fitting import fit_hetero_gp, fit_homo_gp
from kernels import scipy_kernel
from mean_functions import zero_mean
from gp_utils import nlpd, one_d_train_test_split, posterior_predictive


def main(dataset, fplot):
    """
    Benchmark homoscedastic and heteroscedastic Gaussian Process on the datasets from Kersting et al. 2007.

    :param dataset: str specifying the dataset to load. One of:
                    ['scallop', 'silverman', 'yuan', 'williams', 'goldberg', 'lidar']
    :param fplot: bool indicating whether or not to plot the dataset
    """

    Dataset = DatasetLoader(dataset, fplot)
    xs, ys = Dataset.load_data()

    if xs.shape[1] > 1:
        xs_train, xs_test, ys_train, ys_test = train_test_split(xs, ys, test_size=0.1)
    else:
        # This function (permutes the x-values) is needed so that the 1-dimensional datasets get plotted correctly
        xs_train, ys_train, xs_test, ys_test = one_d_train_test_split(xs, ys, split_ratio=0.9)

    # Standardise the outputs

    y_scaler = StandardScaler()
    ys_train = y_scaler.fit_transform(ys_train)

    # Fit the homoscedastic GP

    l_init = 1  # lengthscale to initialise the optimiser with
    sigma_f_init = 1  # signal amplitude to initialise the optimiser with
    noise = 1

    pred_mean, pred_var, nlml = fit_homo_gp(xs_train, ys_train, noise, xs_test, l_init, sigma_f_init, fplot=True)

    # Convert outputs back to real domain before measuring the nlpd.

    pred_var = y_scaler.inverse_transform(pred_var)
    pred_mean = y_scaler.inverse_transform(pred_mean)

    #  will only work if pred_mean has been evaluated at the same positions as the target ys.
    nlpd_val = nlpd(pred_mean, np.diag(pred_var), ys_test)

    print(f'Homoscedastic GP NLPD: {nlpd_val}')

    l_noise_init = 1
    sigma_f_noise_init = 1
    gp2_noise = 1  # We optimise this hyperparameter by hand.
    num_iters = 10
    sample_size = 100

    noise_func, gp2_noise, gp1_l_opt, gp1_sigma_f_opt, gp2_l_opt, gp2_sigma_f_opt, variance_estimator = \
        fit_hetero_gp(xs_train, ys_train, noise, xs_train, l_init, sigma_f_init, l_noise_init, sigma_f_noise_init,
                      gp2_noise, num_iters, sample_size)

    print(gp2_noise)
    print(gp1_l_opt)
    print(gp1_sigma_f_opt)
    print(gp2_l_opt)
    print(gp2_sigma_f_opt)

    pred_mean_het, pred_var_het, _, _ = posterior_predictive(xs_train, ys_train, xs_test, noise_func, gp1_l_opt,
                                                             gp1_sigma_f_opt, mean_func=zero_mean, kernel=scipy_kernel)
    pred_mean_noise, _, _, _ = posterior_predictive(xs_train, variance_estimator, xs_test, gp2_noise, gp2_l_opt,
                                                    gp2_sigma_f_opt, mean_func=zero_mean, kernel=scipy_kernel)

    pred_mean_noise = np.exp(pred_mean_noise).reshape(len(pred_mean_noise))
    pred_var = np.diag(pred_var_het) + pred_mean_noise

    pred_var = y_scaler.inverse_transform(pred_var)
    pred_mean_het = y_scaler.inverse_transform(pred_mean_het)

    nlpd_val_het = nlpd(pred_mean_het, pred_var, ys_test)

    print(nlpd_val_het)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--dataset', type=str, default='silverman',
                        help='Heteroscedastic dataset to generate. Choices are one of '
                             '[lidar, silverman, scallop]')
    parser.add_argument('-p', '--plot', type=bool, default='False',
                        help='Bool indicating whether or not to plot the dataset')

    args = parser.parse_args()

    main(args.dataset, args.plot)

