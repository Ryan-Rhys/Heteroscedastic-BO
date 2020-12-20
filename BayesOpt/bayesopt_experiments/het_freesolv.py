# Copyright Lee Group 2019
# Author: Ryan-Rhys Griffiths
"""
This module contains the code for heteroscedastic Bayesian Optimisation on the Freesolv dataset.
"""

import os
import warnings

from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from sklearn.model_selection import train_test_split

from data_utils import parse_dataset, transform_data
from acquisition_functions import heteroscedastic_one_off_expected_improvement, heteroscedastic_propose_location, \
    my_propose_location, my_expected_improvement, augmented_one_off_expected_improvement, heteroscedastic_one_off_augmented_expected_improvement

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Adjust accordingly for your own file system

FREESOLV_PATH = '../bayesopt_datasets/Freesolv/Freesolv.txt'
task = 'FreeSolv'
use_frag = True
use_exp = True
n_components = 2

if __name__ == '__main__':

    warnings.filterwarnings('ignore')

    xs, ys, std = parse_dataset(task, FREESOLV_PATH, use_frag, use_exp)

    # test in this instane is the initialisation set for Bayesian Optimisation and train is the heldout set.

    xs_train, xs_test, ys_train, ys_test = train_test_split(xs, ys, test_size=0.2, random_state=42, shuffle=False)
    _, _, std_train, std_test = train_test_split(xs, std, test_size=0.2, random_state=42, shuffle=False)
    xs_train, xs_test, ys_train, ys_test, y_scaler = transform_data(xs_train, xs_test, ys_train, ys_test, n_components)

    # Number of iterations
    bayes_opt_iters = 20

    # We perform random trials of Bayesian Optimisation

    homo_running_sum = np.zeros(bayes_opt_iters)
    homo_squares = np.zeros(bayes_opt_iters)  # Following the single-pass estimator given on pg. 192 of mathematics for machine learning
    hetero_running_sum = np.zeros(bayes_opt_iters)
    hetero_squares = np.zeros(bayes_opt_iters)
    aug_running_sum = np.zeros(bayes_opt_iters)
    aug_squares = np.zeros(bayes_opt_iters)
    aug_het_running_sum = np.zeros(bayes_opt_iters)
    aug_het_squares = np.zeros(bayes_opt_iters)

    # We compute the objective corresponding to aleatoric noise only

    homo_noise_running_sum = np.zeros(bayes_opt_iters)
    homo_noise_squares = np.zeros(bayes_opt_iters)  # Following the single-pass estimator given on pg. 192 of mathematics for machine learning
    hetero_noise_running_sum = np.zeros(bayes_opt_iters)
    hetero_noise_squares = np.zeros(bayes_opt_iters)
    aug_noise_running_sum = np.zeros(bayes_opt_iters)
    aug_noise_squares = np.zeros(bayes_opt_iters)
    aug_het_noise_running_sum = np.zeros(bayes_opt_iters)
    aug_het_noise_squares = np.zeros(bayes_opt_iters)

    random_trials = 5

    for i in range(random_trials):

        numpy_seed = i + 62
        tf_seed = i + 63
        np.random.seed(numpy_seed)

        bounds = np.array([np.array([np.min(xs_train[:, i]), np.max(xs_train[:, i])]) for i in range(xs_train.shape[1])])

        if n_components == 2:

            x1_star = np.arange(np.min(xs_train[:, 0]), np.max(xs_train[:, 0]), 0.2)
            x2_star = np.arange(np.min(xs_train[:, 1]), np.max(xs_train[:, 1]), 0.2)

        plot_sample = np.array(np.meshgrid(x1_star, x2_star)).T.reshape(-1, 2)  # Where 2 gives the dimensionality

        X_init = xs_test
        Y_init = ys_test

        # Initialize samples
        homo_X_sample = X_init
        homo_Y_sample = Y_init
        het_X_sample = X_init
        het_Y_sample = Y_init
        aug_X_sample = X_init
        aug_Y_sample = Y_init
        aug_het_X_sample = X_init
        aug_het_Y_sample = Y_init

        # initial BayesOpt hypers

        l_init = 1.0
        sigma_f_init = 1.0
        noise = 1.0  # need to be careful about how we set this because it's not currently being optimised in the code (see reviewer comment)
        l_noise_init = 1.0
        sigma_f_noise_init = 1.0
        gp2_noise = 1.0
        num_iters = 10
        sample_size = 100

        homo_best_so_far = 300  # value to beat
        het_best_so_far = 300
        aug_best_so_far = 300
        aug_het_best_so_far = 300
        homo_obj_val_list = []
        het_obj_val_list = []
        aug_obj_val_list = []
        aug_het_obj_val_list = []
        homo_noise_val_list = []
        het_noise_val_list = []
        aug_noise_val_list = []
        aug_het_noise_val_list = []
        homo_collected_x = []
        het_collected_x = []
        aug_collected_x = []
        aug_het_collected_x = []

        for i in range(bayes_opt_iters):

            print(i)

            # Obtain next sampling point from the acquisition function (expected_improvement)

            homo_X_next = my_propose_location(my_expected_improvement, homo_X_sample, homo_Y_sample, noise, l_init, sigma_f_init,
                                              bounds, plot_sample, n_restarts=3, min_val=300)

            homo_collected_x.append(homo_X_next)

            # Obtain next noisy sample from the objective function
            homo_X_next = min(xs_train, key=lambda x: np.linalg.norm(x - homo_X_next))  # Closest point in the heldout set.
            homo_index = list(xs_train[:, 0]).index(homo_X_next[0])  # index by first dimension
            homo_Y_next = ys_train[homo_index]
            homo_composite_obj_val = homo_Y_next - std_train[homo_index]

            if homo_composite_obj_val < homo_best_so_far:
                homo_best_so_far = homo_composite_obj_val
                homo_obj_val_list.append(homo_composite_obj_val)
            else:
                homo_obj_val_list.append(homo_best_so_far)

            # Add sample to previous samples
            homo_X_sample = np.vstack((homo_X_sample, homo_X_next))
            homo_Y_sample = np.vstack((homo_Y_sample, homo_Y_next))

            # Obtain next sampling point from the het acquisition function (ANPEI)

            het_X_next = heteroscedastic_propose_location(heteroscedastic_one_off_expected_improvement, het_X_sample,
                                                          het_Y_sample, noise, l_init, sigma_f_init, l_noise_init,
                                                          sigma_f_noise_init, gp2_noise, num_iters, sample_size, bounds,
                                                          plot_sample, n_restarts=3, min_val=300)

            het_collected_x.append(het_X_next)

            # Obtain next noisy sample from the objective function
            het_X_next = min(xs_train, key=lambda x: np.linalg.norm(x - het_X_next))
            het_index = list(xs_train[:, 0]).index(het_X_next[0])
            het_Y_next = ys_train[het_index]
            het_composite_obj_val = het_Y_next - std_train[het_index]

            if het_composite_obj_val < het_best_so_far:
                het_best_so_far = het_composite_obj_val
                het_obj_val_list.append(het_composite_obj_val)
            else:
                het_obj_val_list.append(het_best_so_far)

            # Add sample to previous samples
            het_X_sample = np.vstack((het_X_sample, het_X_next))
            het_Y_sample = np.vstack((het_Y_sample, het_Y_next))

            # Obtain next sampling point from the augmented expected improvement (AEI)

            aug_X_next = my_propose_location(augmented_one_off_expected_improvement, aug_X_sample, aug_Y_sample, noise, l_init, sigma_f_init,
                                             bounds, plot_sample, n_restarts=3, min_val=300)

            aug_collected_x.append(aug_X_next)

            # Obtain next noisy sample from the objective function
            aug_X_next = min(xs_train, key=lambda x: np.linalg.norm(x - aug_X_next))
            aug_index = list(xs_train[:, 0]).index(aug_X_next[0])
            aug_Y_next = ys_train[aug_index]
            aug_composite_obj_val = aug_Y_next - std_train[aug_index]

            if aug_composite_obj_val < aug_best_so_far:
                aug_best_so_far = aug_composite_obj_val
                aug_obj_val_list.append(aug_composite_obj_val)
            else:
                aug_obj_val_list.append(aug_best_so_far)

            # Add sample to previous sample
            aug_X_sample = np.vstack((aug_X_sample, aug_X_next))
            aug_Y_sample = np.vstack((aug_Y_sample, aug_Y_next))

            # Obtain next sampling point from the heteroscedastic augmented expected improvement (het-AEI)

            aug_het_X_next = heteroscedastic_propose_location(heteroscedastic_one_off_augmented_expected_improvement, aug_het_X_sample,
                                                          aug_het_Y_sample, noise, l_init, sigma_f_init, l_noise_init,
                                                          sigma_f_noise_init, gp2_noise, num_iters, sample_size, bounds,
                                                          plot_sample, n_restarts=3, min_val=300)

            aug_het_collected_x.append(aug_het_X_next)

            # Obtain next noisy sample from the objective function
            aug_het_X_next = min(xs_train, key=lambda x: np.linalg.norm(x-aug_het_X_next))
            aug_het_index = list(xs_train[:, 0]).index(aug_het_X_next[0])
            aug_het_Y_next = ys_train[aug_het_index]
            aug_het_composite_obj_val = aug_het_Y_next - std_train[aug_het_index]

            if aug_het_composite_obj_val < aug_het_best_so_far:
                aug_het_best_so_far = aug_het_composite_obj_val
                aug_het_obj_val_list.append(aug_het_composite_obj_val)
            else:
                aug_het_obj_val_list.append(aug_het_best_so_far)

            # Add sample to previous sample
            aug_het_X_sample = np.vstack((aug_het_X_sample, aug_het_X_next))
            aug_het_Y_sample = np.vstack((aug_het_Y_sample, aug_het_Y_next))

        homo_running_sum += np.array(homo_obj_val_list, dtype=np.float64).flatten()
        homo_squares += np.array(homo_obj_val_list, dtype=np.float64).flatten() ** 2
        hetero_running_sum += np.array(het_obj_val_list, dtype=np.float64).flatten()
        hetero_squares += np.array(het_obj_val_list, dtype=np.float64).flatten() ** 2
        aug_running_sum += np.array(aug_obj_val_list, dtype=np.float64).flatten()
        aug_squares += np.array(aug_obj_val_list, dtype=np.float64).flatten() ** 2
        aug_het_running_sum += np.array(aug_het_obj_val_list, dtype=np.float64).flatten()
        aug_het_squares += np.array(aug_het_obj_val_list, dtype=np.float64).flatten() ** 2

        # homo_noise_running_sum += np.array(homo_noise_val_list)
        # homo_noise_squares += np.array(homo_noise_val_list) ** 2
        # hetero_noise_running_sum += np.array(het_noise_val_list)
        # hetero_noise_squares += np.array(het_noise_val_list) ** 2

    homo_means = homo_running_sum / random_trials
    hetero_means = hetero_running_sum / random_trials
    homo_errs = np.sqrt(homo_squares / random_trials - homo_means ** 2, dtype=np.float64)
    hetero_errs = np.sqrt(hetero_squares / random_trials - hetero_means ** 2, dtype=np.float64)
    aug_means = aug_running_sum / random_trials
    aug_errs = np.sqrt(aug_squares / random_trials - aug_means ** 2, dtype=np.float64)
    aug_het_means = aug_het_running_sum / random_trials
    aug_het_errs = np.sqrt(aug_het_squares / random_trials - aug_het_means **2, dtype=np.float64)

    print('List of average homoscedastic values is: ' + str(homo_means))
    print('List of homoscedastic errors is: ' + str(homo_errs))
    print('List of average heteroscedastic values is ' + str(hetero_means))
    print('List of heteroscedastic errors is: ' + str(hetero_errs))
    print('List of average AEI values is: ' + str(aug_means))
    print('List of AEI errors is: ' + str(aug_errs))
    print('List of average het-AEI values is: ' + str(aug_het_means))
    print('List of het-AEI errors is: ' + str(aug_het_errs))

    iter_x = np.arange(1, bayes_opt_iters + 1)

    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.plot(iter_x, homo_means, color='r', label='Homoscedastic')
    plt.plot(iter_x, hetero_means, color='b', label='Heteroscedastic ANPEI')
    #plt.plot(iter_x, aug_means, color='g', label='Homoscedastic AEI')
    #plt.plot(iter_x, aug_het_means, color='c', label='Heteroscedastic AEI')
    lower_homo = np.array(homo_means) - np.array(homo_errs)
    upper_homo = np.array(homo_means) + np.array(homo_errs)
    lower_hetero = np.array(hetero_means) - np.array(hetero_errs)
    upper_hetero = np.array(hetero_means) + np.array(hetero_errs)
    lower_aei = np.array(aug_means) - np.array(aug_errs)
    upper_aei = np.array(aug_means) + np.array(aug_errs)
    lower_het_aei = np.array(aug_het_means) - np.array(aug_het_errs)
    upper_het_aei = np.array(aug_het_means) + np.array(aug_het_errs)
    plt.fill_between(iter_x, lower_homo, upper_homo, color='r', label='Homoscedastic', alpha=0.1)
    plt.fill_between(iter_x, lower_hetero, upper_hetero, color='b', label='Heteroscedastic ANPEI', alpha=0.1)
    #plt.fill_between(iter_x, lower_aei, upper_aei, color='g', label='Homoscedastic AEI', alpha=0.1)
    #plt.fill_between(iter_x, lower_het_aei, upper_het_aei, color='c', label='Heteroscedastic AEI', alpha=0.1)
    plt.title('Best Objective Function Value Found so Far')
    plt.xlabel('Number of Function Evaluations')
    plt.ylabel('Objective Function Value - Noise')
    plt.legend(loc=1)
    plt.savefig('real_datasets_figures/bayesopt_plot{}_iters_{}_random_trials_and_seed_{}_with_het_aei_full_unc_just_tw_robust_new'.format(bayes_opt_iters, random_trials, numpy_seed))
