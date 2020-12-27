# Author: Ryan-Rhys Griffiths
"""
This script benchmarks homoscedastic and heteroscedastic GPs on the datasets from Kersting et al. in terms of
their negative log predictive density (NLPD) on heldout test points.
"""

import argparse

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from dataset_loaders import DatasetLoader
from gp_fitting import fit_hetero_gp, fit_homo_gp
from kernels import scipy_kernel
from mean_functions import zero_mean
from gp_utils import nlpd, one_d_train_test_split, posterior_predictive

# Hyperparameter initialisation settings to reproduce the results of Kersting et al. 2007

# lidar: {l_init: 50, sigma_f_init: 0.1, noise: 0.2, gp2_l_init: 0.1, gp2_sigma_f_init: 0.3, gp2_noise: 0.2}
# williams: {l_init: 0.5 , sigma_f_init: 1.2 , noise: 0.2 , gp2_l_init: 1, gp2_sigma_f_init: 2, gp2_noise: 0.5}
# yuan: {l_init: , sigma_f_init: , noise: , gp2_l_init: , gp2_sigma_f_init: , gp2_noise: }
# silverman: {l_init: , sigma_f_init: , noise: , gp2_l_init: , gp2_sigma_f_init: , gp2_noise: }
# goldberg: {l_init: , sigma_f_init: , noise: , gp2_l_init: , gp2_sigma_f_init: , gp2_noise: }
# scallop: {l_init: , sigma_f_init: , noise: , gp2_l_init: , gp2_sigma_f_init: , gp2_noise: }


def main(dataset, fplot, n_trials):
    """
    Benchmark homoscedastic and heteroscedastic Gaussian Process on the datasets from Kersting et al. 2007.

    :param dataset: str specifying the dataset to load. One of:
                    ['scallop', 'silverman', 'yuan', 'williams', 'goldberg', 'lidar']
    :param fplot: bool indicating whether or not to plot the dataset
    :param n_trials: The number of random trials. Defaults to 10 as used in Kersting et al. 2007
    """

    Dataset = DatasetLoader(dataset, fplot)
    xs, ys = Dataset.load_data()

    # Initialise MSE and NLPD lists to store the values for the random trials

    mse_list_hom = []
    nlpd_list_hom = []
    mse_list_het = []
    nlpd_list_het = []

    for i in range(n_trials):

        if xs.shape[1] > 1:
            xs_train, xs_test, ys_train, ys_test = train_test_split(xs, ys, test_size=0.1, random_state=i)
        else:
            # This function (permutes the x-values) is needed so that the 1-dimensional datasets get plotted correctly
            xs_train, ys_train, xs_test, ys_test = one_d_train_test_split(xs, ys, split_ratio=0.9, seed=i)

        # Fit the homoscedastic GP

        l_init = 0.1  # lengthscale to initialise the optimiser with
        sigma_f_init = 3  # signal amplitude to initialise the optimiser with
        noise = 0.1  # noise to initialise the optimiser with. Same for both homoscedastic GP and GP1 of MLHGP

        pred_mean, pred_var, nlml = fit_homo_gp(xs_train, ys_train, noise, xs_test, l_init, sigma_f_init, fplot=True)

        mse_val_hom = mean_squared_error(pred_mean, ys_test)
        #  will only work if pred_mean has been evaluated at the same positions as the target ys.
        nlpd_val_hom = nlpd(pred_mean, np.diag(pred_var), ys_test)

        mse_list_hom.append(mse_val_hom)
        nlpd_list_hom.append(nlpd_val_hom)

        print(f'Homoscedastic GP MSE for trial {i}: {mse_val_hom}')
        print(f'Homoscedastic GP NLPD for trial {i}: {nlpd_val_hom}')

        gp2_l_init = 0.3
        gp2_sigma_f_init = 2.5
        gp2_noise = 0.5
        num_iters = 10
        sample_size = 100

        noise_func, gp2_noise, gp1_l_opt, gp1_sigma_f_opt, gp2_l_opt, gp2_sigma_f_opt, variance_estimator = \
            fit_hetero_gp(xs_train, ys_train, noise, xs_train, l_init, sigma_f_init, gp2_l_init, gp2_sigma_f_init,
                          gp2_noise, num_iters, sample_size)

        het_gp_param_dict = {'GP1 lengthscale': gp1_l_opt, 'GP1 signal amplitude': gp1_sigma_f_opt, 'GP1 noise': noise,
                             'GP2 lengthscale': gp2_l_opt, 'GP2 signal amplitude': gp2_sigma_f_opt, 'GP2 noise': gp2_noise}

        print('Optimised Hyperparameters for MLHGP: \n')
        print(het_gp_param_dict)

        pred_mean_het, pred_var_het, _, _ = posterior_predictive(xs_train, ys_train, xs_test, noise_func, gp1_l_opt,
                                                                 gp1_sigma_f_opt, mean_func=zero_mean, kernel=scipy_kernel)
        pred_mean_noise, _, _, _ = posterior_predictive(xs_train, variance_estimator, xs_test, gp2_noise, gp2_l_opt,
                                                        gp2_sigma_f_opt, mean_func=zero_mean, kernel=scipy_kernel)

        pred_mean_noise = np.exp(pred_mean_noise).reshape(len(pred_mean_noise))
        pred_var = np.diag(pred_var_het) + pred_mean_noise

        nlpd_val_het = nlpd(pred_mean_het, pred_var, ys_test)
        mse_val_het = mean_squared_error(pred_mean_het, ys_test)

        mse_list_het.append(mse_val_het)
        nlpd_list_het.append(nlpd_val_het)

        print(f'Heteroscedastic GP MSE for trial {i}: {mse_val_het}')
        print(f'Heteroscedastic GP NLPD for trial {i}: {nlpd_val_het}')

    avg_mse_hom = np.mean(mse_list_hom)
    avg_mse_het = np.mean(mse_list_het)
    std_mse_hom = np.std(mse_list_hom) / np.sqrt(n_trials)
    std_mse_het = np.std(mse_list_het) / np.sqrt(n_trials)

    avg_nlpd_hom = np.mean(nlpd_list_hom)
    avg_nlpd_het = np.mean(nlpd_list_het)
    std_nlpd_hom = np.std(nlpd_list_hom) / np.sqrt(n_trials)
    std_nlpd_het = np.std(nlpd_list_het) / np.sqrt(n_trials)

    print(f'Mean MSE for homoscedatic GP is {avg_mse_hom} with standard error of {std_mse_hom}')
    print(f'Mean NLPD for homoscedatic GP is {avg_nlpd_hom} with standard error of {std_nlpd_hom}')
    print(f'Mean MSE for heteroscedastic GP is {avg_mse_het} with standard error of {std_mse_het}')
    print(f'Mean NLPD for heteroscedastic GP is {avg_nlpd_het} with standard error of {std_nlpd_het}')

    # Save the MSE and NLPD values.

    results_dict = {'hom_mean_mse': avg_mse_hom, 'hom_std_mse': std_mse_hom, 'hom_mean_nlpd': avg_nlpd_hom,
                    'hom_std_nlpd': std_nlpd_hom, 'het_mean_mse': avg_mse_het, 'het_std_mse': std_mse_het,
                    'het_mean_nlpd': avg_nlpd_het, 'het_std_nlpd': std_nlpd_het}

    with open(f'results/{dataset}/mse_nlpd_{n_trials}.txt', 'w') as f:
        f.write(str(results_dict))
        f.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--dataset', type=str, default='yuan',
                        help='Heteroscedastic dataset to generate. Choices are one of '
                             '[lidar, silverman, scallop, yuan, williams, goldberg]')
    parser.add_argument('-p', '--fplot', type=bool, default=False,
                        help='Bool indicating whether or not to plot the dataset')
    parser.add_argument('-n', '--n_trials', type=int, default=10,
                        help='The number of random trials')

    args = parser.parse_args()

    main(args.dataset, args.fplot, args.n_trials)
