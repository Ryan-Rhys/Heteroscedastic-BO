# Author: Ryan-Rhys Griffiths
"""
This module contains the code for heteroscedastic Bayesian Optimisation on the Freesolv dataset.
"""

import argparse
import os
import sys
import warnings

from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

sys.path.append('../')
sys.path.append('../..')
sys.path.append('../../..')

from data_utils import parse_dataset
from acquisition_funcs.acquisition_functions import heteroscedastic_expected_improvement, heteroscedastic_propose_location, \
    my_propose_location, my_expected_improvement, augmented_expected_improvement, heteroscedastic_augmented_expected_improvement

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def main(penalty, aleatoric_weight, random_trials, bayes_opt_iters, init_set_size, n_components, path):
    """
    Script for running the soil phosphorus fraction optimisation experiment.

    param: penalty: $\alpha$ parameter specifying weight of noise component to objective
    param: aleatoric_weight: float specifying the value of $\beta of ANPEI
    param: random_trials: int specifying the number of random initialisations
    param: bayes_opt_iters: int specifying the number of iterations of BayesOpt
    param: init_set_size: int specifying the side length of the 2D grid to initialise on.
    param: n_components: int specifying the number of PCA principle components to keep.
    param: path: str specifying the path to the Freesolv.txt file.
    """

    task = 'FreeSolv'
    use_frag = True
    use_exp = True  # use experimental values.

    xs, ys, std = parse_dataset(task, path, use_frag, use_exp)

    warnings.filterwarnings('ignore')

    # Number of iterations
    bayes_opt_iters = 10
    random_trials = 50

    # We perform random trials of Bayesian Optimisation

    rand_running_sum = np.zeros(bayes_opt_iters)
    rand_squares = np.zeros(bayes_opt_iters)
    homo_running_sum = np.zeros(bayes_opt_iters)
    homo_squares = np.zeros(bayes_opt_iters)  # Following the single-pass estimator given on pg. 192 of mathematics for machine learning
    hetero_running_sum = np.zeros(bayes_opt_iters)
    hetero_squares = np.zeros(bayes_opt_iters)
    aug_running_sum = np.zeros(bayes_opt_iters)
    aug_squares = np.zeros(bayes_opt_iters)
    aug_het_running_sum = np.zeros(bayes_opt_iters)
    aug_het_squares = np.zeros(bayes_opt_iters)

    # We compute the objective corresponding to aleatoric noise only

    rand_noise_running_sum = np.zeros(bayes_opt_iters)
    rand_noise_squares = np.zeros(bayes_opt_iters)
    homo_noise_running_sum = np.zeros(bayes_opt_iters)
    homo_noise_squares = np.zeros(bayes_opt_iters)  # Following the single-pass estimator given on pg. 192 of mathematics for machine learning
    hetero_noise_running_sum = np.zeros(bayes_opt_iters)
    hetero_noise_squares = np.zeros(bayes_opt_iters)
    aug_noise_running_sum = np.zeros(bayes_opt_iters)
    aug_noise_squares = np.zeros(bayes_opt_iters)
    aug_het_noise_running_sum = np.zeros(bayes_opt_iters)
    aug_het_noise_squares = np.zeros(bayes_opt_iters)

    for i in range(random_trials):

        start_seed = 47
        numpy_seed = i + start_seed  # set to avoid segfault issue
                             # ('Process finished with exit code 139 (interrupted by signal 11: SIGSEGV)') when i = 0

        # test in this instance is the initialisation set for Bayesian Optimisation and train is the heldout set.

        xs_train, xs_test, ys_train, ys_test = train_test_split(xs, ys, test_size=init_set_size, random_state=numpy_seed, shuffle=True)

        pca = PCA(n_components)
        xs_test = pca.fit_transform(xs_test)
        print('Fraction of variance retained is: ' + str(sum(pca.explained_variance_ratio_)))
        xs_train = pca.transform(xs_train)

        _, _, std_train, std_test = train_test_split(xs, std, test_size=init_set_size, random_state=numpy_seed, shuffle=True)

        ys_train = ys_train.reshape(-1, 1)
        ys_test = ys_test.reshape(-1, 1)

        init_num_samples = len(ys_test)

        bounds = np.array([np.array([np.min(xs_train[:, i]), np.max(xs_train[:, i])]) for i in range(xs_train.shape[1])])

        # Can only plot in 2D

        if n_components == 2:
            x1_star = np.arange(np.min(xs_train[:, 0]), np.max(xs_train[:, 0]), 0.2)
            x2_star = np.arange(np.min(xs_train[:, 1]), np.max(xs_train[:, 1]), 0.2)
            plot_sample = np.array(np.meshgrid(x1_star, x2_star)).T.reshape(-1, 2)  # Where 2 gives the dimensionality
        else:
            plot_sample = None

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

        # initial GP hypers

        l_init = 1.0
        sigma_f_init = 1.0
        noise = 1.0
        l_noise_init = 1.0
        sigma_f_noise_init = 1.0
        gp2_noise = 1.0
        num_iters = 10
        sample_size = 100

        rand_best_so_far = 300
        homo_best_so_far = 300  # value to beat
        het_best_so_far = 300
        aug_best_so_far = 300
        aug_het_best_so_far = 300
        rand_noise_best_so_far = 300
        homo_noise_best_so_far = 300  # value to beat
        het_noise_best_so_far = 300
        aug_noise_best_so_far = 300
        aug_het_noise_best_so_far = 300
        rand_obj_val_list = []
        homo_obj_val_list = []
        het_obj_val_list = []
        aug_obj_val_list = []
        aug_het_obj_val_list = []
        rand_noise_val_list = []
        homo_noise_val_list = []
        het_noise_val_list = []
        aug_noise_val_list = []
        aug_het_noise_val_list = []
        rand_collected_x = []
        homo_collected_x = []
        het_collected_x = []
        aug_collected_x = []
        aug_het_collected_x = []

        for j in range(bayes_opt_iters):

            print(j)

            # take random point from uniform distribution
            rand_X_next = np.random.uniform(np.min(xs_train, axis=0), np.max(xs_train, axis=0))  # this just takes X not the sin function itself
            # Obtain next noisy sample from the objective function
            rand_X_next = min(xs_train, key=lambda x: np.linalg.norm(x - rand_X_next))  # Closest point in the heldout set.
            rand_index = list(xs_train[:, 0]).index(rand_X_next[0])  # index by first dimension
            rand_Y_next = ys_train[rand_index]
            rand_composite_obj_val = rand_Y_next + penalty*std_train[rand_index]
            rand_noise_val = std_train[rand_index]
            rand_collected_x.append(rand_X_next)

            # check if random point's Y value is better than best so far
            if rand_composite_obj_val < rand_best_so_far:
                rand_best_so_far = rand_composite_obj_val
                rand_obj_val_list.append(rand_composite_obj_val)
            else:
                rand_obj_val_list.append(rand_best_so_far)
            # if yes, save it, if no, save best so far into list of best y-value per iteration in rand_composite_obj_val

            if rand_noise_val < rand_noise_best_so_far:
                rand_noise_best_so_far = rand_noise_val
                rand_noise_val_list.append(rand_noise_val)
            else:
                rand_noise_val_list.append(rand_noise_best_so_far)

            # Obtain next sampling point from the acquisition function (expected_improvement)

            homo_X_next = my_propose_location(my_expected_improvement, homo_X_sample, homo_Y_sample, noise, l_init, sigma_f_init,
                                              bounds, plot_sample, n_restarts=3, min_val=300)

            homo_collected_x.append(homo_X_next)

            # Obtain next noisy sample from the objective function
            homo_X_next = min(xs_train, key=lambda x: np.linalg.norm(x - homo_X_next))  # Closest point in the heldout set.
            homo_index = list(xs_train[:, 0]).index(homo_X_next[0])  # index by first dimension
            homo_Y_next = ys_train[homo_index]
            homo_composite_obj_val = homo_Y_next + penalty*std_train[homo_index]
            homo_noise_val = std_train[homo_index]
            homo_collected_x.append(homo_X_next)

            if homo_composite_obj_val < homo_best_so_far:
                homo_best_so_far = homo_composite_obj_val
                homo_obj_val_list.append(homo_composite_obj_val)
            else:
                homo_obj_val_list.append(homo_best_so_far)

            if homo_noise_val < homo_noise_best_so_far:
                homo_noise_best_so_far = homo_noise_val
                homo_noise_val_list.append(homo_noise_val)
            else:
                homo_noise_val_list.append(homo_noise_best_so_far)

            # Add sample to previous samples
            homo_X_sample = np.vstack((homo_X_sample, homo_X_next))
            homo_Y_sample = np.vstack((homo_Y_sample, homo_Y_next))

            # Obtain next sampling point from the het acquisition function (ANPEI)

            het_X_next = heteroscedastic_propose_location(heteroscedastic_expected_improvement, het_X_sample,
                                                          het_Y_sample, noise, l_init, sigma_f_init, l_noise_init,
                                                          sigma_f_noise_init, gp2_noise, num_iters, sample_size, bounds,
                                                          plot_sample, n_restarts=3, min_val=300, aleatoric_weight=aleatoric_weight)

            het_collected_x.append(het_X_next)

            # Obtain next noisy sample from the objective function
            het_X_next = min(xs_train, key=lambda x: np.linalg.norm(x - het_X_next))
            het_index = list(xs_train[:, 0]).index(het_X_next[0])
            het_Y_next = ys_train[het_index]
            het_composite_obj_val = het_Y_next + penalty*std_train[het_index]
            het_noise_val = std_train[het_index]
            het_collected_x.append(het_X_next)

            if het_composite_obj_val < het_best_so_far:
                het_best_so_far = het_composite_obj_val
                het_obj_val_list.append(het_composite_obj_val)
            else:
                het_obj_val_list.append(het_best_so_far)

            if het_noise_val < het_noise_best_so_far:
                het_noise_best_so_far = het_noise_val
                het_noise_val_list.append(het_noise_val)
            else:
                het_noise_val_list.append(het_noise_best_so_far)

            # Add sample to previous samples
            het_X_sample = np.vstack((het_X_sample, het_X_next))
            het_Y_sample = np.vstack((het_Y_sample, het_Y_next))

            # Obtain next sampling point from the augmented expected improvement (AEI)

            aug_X_next = my_propose_location(augmented_expected_improvement, aug_X_sample, aug_Y_sample, noise, l_init, sigma_f_init,
                                             bounds, plot_sample, n_restarts=3, min_val=300, aleatoric_weight=aleatoric_weight, aei=True)

            aug_collected_x.append(aug_X_next)

            # Obtain next noisy sample from the objective function
            aug_X_next = min(xs_train, key=lambda x: np.linalg.norm(x - aug_X_next))
            aug_index = list(xs_train[:, 0]).index(aug_X_next[0])
            aug_Y_next = ys_train[aug_index]
            aug_composite_obj_val = aug_Y_next + penalty*std_train[aug_index]
            aug_noise_val = std_train[aug_index]
            aug_collected_x.append(het_X_next)

            if aug_composite_obj_val < aug_best_so_far:
                aug_best_so_far = aug_composite_obj_val
                aug_obj_val_list.append(aug_composite_obj_val)
            else:
                aug_obj_val_list.append(aug_best_so_far)

            if aug_noise_val < aug_noise_best_so_far:
                aug_noise_best_so_far = aug_noise_val
                aug_noise_val_list.append(aug_noise_val)
            else:
                aug_noise_val_list.append(aug_noise_best_so_far)

            # Add sample to previous sample
            aug_X_sample = np.vstack((aug_X_sample, aug_X_next))
            aug_Y_sample = np.vstack((aug_Y_sample, aug_Y_next))

            # Obtain next sampling point from the heteroscedastic augmented expected improvement (het-AEI)

            aug_het_X_next = heteroscedastic_propose_location(heteroscedastic_augmented_expected_improvement, aug_het_X_sample,
                                                          aug_het_Y_sample, noise, l_init, sigma_f_init, l_noise_init,
                                                          sigma_f_noise_init, gp2_noise, num_iters, sample_size, bounds,
                                                          plot_sample, n_restarts=3, min_val=300, aleatoric_weight=aleatoric_weight)

            aug_het_collected_x.append(aug_het_X_next)

            # Obtain next noisy sample from the objective function
            aug_het_X_next = min(xs_train, key=lambda x: np.linalg.norm(x - aug_het_X_next))
            aug_het_index = list(xs_train[:, 0]).index(aug_het_X_next[0])
            aug_het_Y_next = ys_train[aug_het_index]
            aug_het_composite_obj_val = aug_het_Y_next + penalty*std_train[aug_het_index]
            aug_het_noise_val = std_train[aug_het_index]
            aug_het_collected_x.append(aug_het_X_next)

            if aug_het_composite_obj_val < aug_het_best_so_far:
                aug_het_best_so_far = aug_het_composite_obj_val
                aug_het_obj_val_list.append(aug_het_composite_obj_val)
            else:
                aug_het_obj_val_list.append(aug_het_best_so_far)

            if aug_het_noise_val < aug_het_noise_best_so_far:
                aug_het_noise_best_so_far = aug_het_noise_val
                aug_het_noise_val_list.append(aug_het_noise_val)
            else:
                aug_het_noise_val_list.append(aug_het_noise_best_so_far)

            # Add sample to previous sample
            aug_het_X_sample = np.vstack((aug_het_X_sample, aug_het_X_next))
            aug_het_Y_sample = np.vstack((aug_het_Y_sample, aug_het_Y_next))

        rand_running_sum += np.array(rand_obj_val_list, dtype=np.float64).flatten()
        rand_squares += np.array(rand_obj_val_list, dtype=np.float64).flatten() ** 2
        homo_running_sum += np.array(homo_obj_val_list, dtype=np.float64).flatten()
        homo_squares += np.array(homo_obj_val_list, dtype=np.float64).flatten() ** 2
        hetero_running_sum += np.array(het_obj_val_list, dtype=np.float64).flatten()
        hetero_squares += np.array(het_obj_val_list, dtype=np.float64).flatten() ** 2
        aug_running_sum += np.array(aug_obj_val_list, dtype=np.float64).flatten()
        aug_squares += np.array(aug_obj_val_list, dtype=np.float64).flatten() ** 2
        aug_het_running_sum += np.array(aug_het_obj_val_list, dtype=np.float64).flatten()
        aug_het_squares += np.array(aug_het_obj_val_list, dtype=np.float64).flatten() ** 2

        rand_noise_running_sum += np.array(rand_noise_val_list, dtype=np.float64).flatten()  # just the way to average out across all random trials
        rand_noise_squares += np.array(rand_noise_val_list, dtype=np.float64).flatten() ** 2  # likewise for errors
        homo_noise_running_sum += np.array(homo_noise_val_list, dtype=np.float64).flatten()
        homo_noise_squares += np.array(homo_noise_val_list, dtype=np.float64).flatten() ** 2
        hetero_noise_running_sum += np.array(het_noise_val_list, dtype=np.float64).flatten()
        hetero_noise_squares += np.array(het_noise_val_list, dtype=np.float64).flatten() ** 2
        aug_noise_running_sum += np.array(aug_noise_val_list, dtype=np.float64).flatten()
        aug_noise_squares += np.array(aug_noise_val_list, dtype=np.float64).flatten() ** 2
        aug_het_noise_running_sum += np.array(aug_het_noise_val_list, dtype=np.float64).flatten()
        aug_het_noise_squares += np.array(aug_het_noise_val_list, dtype=np.float64).flatten() ** 2

        print(f'trial {i} complete')

        if init_set_size == 0.2:

            seed_index = i + start_seed + 1

            np.savetxt(f'freesolv_data/02_pen_1/rand_means/rand_means_{start_seed}_{seed_index}.txt', rand_running_sum)
            np.savetxt(f'freesolv_data/02_pen_1/rand_means/rand_squares_{start_seed}_{seed_index}.txt', rand_squares)
            np.savetxt(f'freesolv_data/02_pen_1/homo_means/homo_means_{start_seed}_{seed_index}.txt', homo_running_sum)
            np.savetxt(f'freesolv_data/02_pen_1/homo_means/homo_squares_{start_seed}_{seed_index}.txt', homo_squares)
            np.savetxt(f'freesolv_data/02_pen_1/het_means/hetero_means_{start_seed}_{seed_index}.txt', hetero_running_sum)
            np.savetxt(f'freesolv_data/02_pen_1/het_means/hetero_squares_{start_seed}_{seed_index}.txt', hetero_squares)
            np.savetxt(f'freesolv_data/02_pen_1/aug_means/aug_means_{start_seed}_{seed_index}.txt', aug_running_sum)
            np.savetxt(f'freesolv_data/02_pen_1/aug_means/aug_squares_{start_seed}_{seed_index}.txt', aug_squares)
            np.savetxt(f'freesolv_data/02_pen_1/aug_het_means/aug_het_means_{start_seed}_{seed_index}.txt', aug_het_running_sum)
            np.savetxt(f'freesolv_data/02_pen_1/aug_het_means/aug_het_squares_{start_seed}_{seed_index}.txt', aug_het_squares)

            np.savetxt(f'freesolv_data/02_pen_1/rand_noise/rand_means_{start_seed}_{seed_index}.txt', rand_noise_running_sum)
            np.savetxt(f'freesolv_data/02_pen_1/rand_noise/rand_squares_{start_seed}_{seed_index}.txt', rand_noise_squares)
            np.savetxt(f'freesolv_data/02_pen_1/homo_noise/homo_means_{start_seed}_{seed_index}.txt', homo_noise_running_sum)
            np.savetxt(f'freesolv_data/02_pen_1/homo_noise/homo_squares_{start_seed}_{seed_index}.txt', homo_noise_squares)
            np.savetxt(f'freesolv_data/02_pen_1/het_noise/hetero_means_{start_seed}_{seed_index}.txt', hetero_noise_running_sum)
            np.savetxt(f'freesolv_data/02_pen_1/het_noise/hetero_squares_{start_seed}_{seed_index}.txt', hetero_noise_squares)
            np.savetxt(f'freesolv_data/02_pen_1/aug_noise/aug_means_{start_seed}_{seed_index}.txt', aug_noise_running_sum)
            np.savetxt(f'freesolv_data/02_pen_1/aug_noise/aug_squares_{start_seed}_{seed_index}.txt', aug_noise_squares)
            np.savetxt(f'freesolv_data/02_pen_1/aug_het_noise/aug_het_means_{start_seed}_{seed_index}.txt', aug_het_noise_running_sum)
            np.savetxt(f'freesolv_data/02_pen_1/aug_het_noise/aug_het_squares_{start_seed}_{seed_index}.txt', aug_het_noise_squares)

    rand_means = rand_running_sum / random_trials
    rand_errs = (np.sqrt(rand_squares / random_trials - rand_means **2))/np.sqrt(random_trials)
    homo_means = homo_running_sum / random_trials
    hetero_means = hetero_running_sum / random_trials
    homo_errs = (np.sqrt(homo_squares / random_trials - homo_means ** 2, dtype=np.float64))/np.sqrt(random_trials)
    hetero_errs = (np.sqrt(hetero_squares / random_trials - hetero_means ** 2, dtype=np.float64))/np.sqrt(random_trials)
    aug_means = aug_running_sum / random_trials
    aug_errs = (np.sqrt(aug_squares / random_trials - aug_means ** 2, dtype=np.float64))/np.sqrt(random_trials)
    aug_het_means = aug_het_running_sum / random_trials
    aug_het_errs = (np.sqrt(aug_het_squares / random_trials - aug_het_means **2, dtype=np.float64))/np.sqrt(random_trials)

    rand_noise_means = rand_noise_running_sum / random_trials
    homo_noise_means = homo_noise_running_sum / random_trials
    hetero_noise_means = hetero_noise_running_sum / random_trials
    rand_noise_errs = (np.sqrt(rand_noise_squares / random_trials - rand_noise_means ** 2))/np.sqrt(random_trials)
    homo_noise_errs = (np.sqrt(homo_noise_squares / random_trials - homo_noise_means ** 2))/np.sqrt(random_trials)
    hetero_noise_errs = (np.sqrt(hetero_noise_squares / random_trials - hetero_noise_means ** 2))/np.sqrt(random_trials)
    aug_noise_means = aug_noise_running_sum / random_trials
    aug_noise_errs = (np.sqrt(aug_noise_squares / random_trials - aug_noise_means ** 2))/np.sqrt(random_trials)
    aug_het_noise_means = aug_het_noise_running_sum / random_trials
    aug_het_noise_errs = (np.sqrt(aug_het_noise_squares / random_trials - aug_het_noise_means ** 2))/np.sqrt(random_trials)

    print('List of average random values is: ' + str(rand_means))
    print('List of random errors is: ' + str(rand_noise_means))
    print('List of average homoscedastic values is: ' + str(homo_means))
    print('List of homoscedastic errors is: ' + str(homo_noise_means))
    print('List of average heteroscedastic values is ' + str(hetero_means))
    print('List of heteroscedastic errors is: ' + str(hetero_noise_means))
    print('List of average AEI values is: ' + str(aug_means))
    print('List of AEI errors is: ' + str(aug_noise_means))
    print('List of average het-AEI values is: ' + str(aug_het_means))
    print('List of het-AEI errors is: ' + str(aug_het_noise_means))

    iter_x = np.arange(1, bayes_opt_iters + 1)

    # clear figure from previous fplot returns if fiddling with form of function
    plt.cla()

    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    lower_rand = np.array(rand_means) - np.array(rand_errs)
    upper_rand = np.array(rand_means) + np.array(rand_errs)
    lower_homo = np.array(homo_means) - np.array(homo_errs)
    upper_homo = np.array(homo_means) + np.array(homo_errs)
    lower_hetero = np.array(hetero_means) - np.array(hetero_errs)
    upper_hetero = np.array(hetero_means) + np.array(hetero_errs)
    lower_aei = np.array(aug_means) - np.array(aug_errs)
    upper_aei = np.array(aug_means) + np.array(aug_errs)
    lower_het_aei = np.array(aug_het_means) - np.array(aug_het_errs)
    upper_het_aei = np.array(aug_het_means) + np.array(aug_het_errs)

    plt.plot(iter_x, rand_means, color='tab:orange', label='RS')
    plt.plot(iter_x, homo_means, color='tab:blue', label='EI')
    plt.plot(iter_x, hetero_means, color='tab:green', label='ANPEI')
    plt.plot(iter_x, aug_means, color='tab:red', label='AEI')
    plt.plot(iter_x, aug_het_means, color='tab:purple', label='HAEI')
    plt.fill_between(iter_x, lower_rand, upper_rand, color='tab:orange', alpha=0.1)
    plt.fill_between(iter_x, lower_homo, upper_homo, color='tab:blue', alpha=0.1)
    plt.fill_between(iter_x, lower_hetero, upper_hetero, color='tab:green', alpha=0.1)
    plt.fill_between(iter_x, lower_aei, upper_aei, color='tab:red', alpha=0.1)
    plt.fill_between(iter_x, lower_het_aei, upper_het_aei, color='tab:purple', alpha=0.1)

    #plt.title('Best Objective Function Value Found so Far', fontsize=16)
    plt.xlabel('Function Evaluations', fontsize=14)
    if penalty != 1:
        plt.ylabel(f'Hydration Free Energy (kcal/mol) + {penalty}*Noise', fontsize=14)
    else:
        plt.ylabel(f'Hydration Free Energy (kcal/mol) + Noise', fontsize=14)
    plt.tick_params(labelsize=14)
    #plt.legend(loc=1)
    plt.legend(loc='lower left', bbox_to_anchor=(0.0, -0.425), ncol=3, borderaxespad=0, fontsize=14, frameon=False)
    plt.savefig('new_freesolv_figures/bayesopt_plot{}_iters_{}_random_trials_and_init_num_samples_of_{}_and_seed_{}_'
                'new_acq_penalty_is_{}_aleatoric_weight_is_{}_n_components_is_{}_new_aei_comp_seed_check'.
                format(bayes_opt_iters, random_trials, init_num_samples, numpy_seed, penalty, aleatoric_weight, n_components), bbox_inches='tight')

    plt.close()

    # clear figure from previous fplot returns if fiddling with form of function
    plt.cla()

    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    lower_noise_rand = np.array(rand_noise_means) - np.array(rand_noise_errs)
    upper_noise_rand = np.array(rand_noise_means) + np.array(rand_noise_errs)
    lower_noise_homo = np.array(homo_noise_means) - np.array(homo_noise_errs)
    upper_noise_homo = np.array(homo_noise_means) + np.array(homo_noise_errs)
    lower_noise_hetero = np.array(hetero_noise_means) - np.array(hetero_noise_errs)
    upper_noise_hetero = np.array(hetero_noise_means) + np.array(hetero_noise_errs)
    lower_noise_aei = np.array(aug_noise_means) - np.array(aug_noise_errs)
    upper_noise_aei = np.array(aug_noise_means) + np.array(aug_noise_errs)
    lower_noise_het_aei = np.array(aug_het_noise_means) - np.array(aug_het_noise_errs)
    upper_noise_het_aei = np.array(aug_het_noise_means) + np.array(aug_het_noise_errs)

    plt.plot(iter_x, rand_noise_means, color='tab:orange', label='RS')
    plt.plot(iter_x, homo_noise_means, color='tab:blue', label='EI')
    plt.plot(iter_x, hetero_noise_means, color='tab:green', label='ANPEI')
    plt.plot(iter_x, aug_noise_means, color='tab:red', label='AEI')
    plt.plot(iter_x, aug_het_noise_means, color='tab:purple', label='HAEI')
    plt.fill_between(iter_x, lower_noise_rand, upper_noise_rand, color='tab:orange', alpha=0.1)
    plt.fill_between(iter_x, lower_noise_homo, upper_noise_homo, color='tab:blue', alpha=0.1)
    plt.fill_between(iter_x, lower_noise_hetero, upper_noise_hetero, color='tab:green', alpha=0.1)
    plt.fill_between(iter_x, lower_noise_aei, upper_noise_aei, color='tab:red', alpha=0.1)
    plt.fill_between(iter_x, lower_noise_het_aei, upper_noise_het_aei, color='tab:purple', alpha=0.1)

    #plt.title('Lowest Aleatoric Noise Found so Far', fontsize=16)
    plt.xlabel('Function Evaluations', fontsize=14)
    plt.ylabel('Aleatoric Noise', fontsize=14)
    plt.tick_params(labelsize=14)
    #plt.legend(loc=1)
    plt.legend(loc='lower left', bbox_to_anchor=(0.0, -0.425), ncol=3, borderaxespad=0, fontsize=14, frameon=False)
    plt.savefig('new_freesolv_figures/bayesopt_plot{}_iters_{}_random_trials_and_init_num_samples_of_{}_and_seed_{}_'
                'noise_only_new_acq_penalty_is_{}_aleatoric_weight_is_{}_n_components_is_{}_new_aei_comp_seed_check'.
                format(bayes_opt_iters, random_trials, init_num_samples, numpy_seed, penalty, aleatoric_weight, n_components), bbox_inches='tight')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--penalty', type=int, default=1,
                        help='$\alpha$ parameter specifying weight of noise component to objective.')
    parser.add_argument('-a', '--aleatoric_weight', type=float, default=1,
                        help='The value of both $\beta and $\gamma of ANPEI and HAEI')
    parser.add_argument('-r', '--random_trials', type=int, default=50,
                        help='Number of random initialisations')
    parser.add_argument('-b', '--bayes_opt_iters', type=int, default=10,
                        help='The number of iterations of BayesOpt')
    parser.add_argument('-t', '--init_set_size', type=float, default=0.2,
                        help='The fraction of datapoints to initialise with')
    parser.add_argument('-pc', '--n_components', type=int, default=14,
                        help='The number of principle components to keep')
    parser.add_argument('-path', '--path', type=str, default='../bayesopt_datasets/Freesolv/Freesolv.txt',
                        help='The path to the Freesolv.txt file')

    args = parser.parse_args()

    main(args.penalty, args.aleatoric_weight, args.random_trials, args.bayes_opt_iters, args.init_set_size,
         args.n_components, args.path)
