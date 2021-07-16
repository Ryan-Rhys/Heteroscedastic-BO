# Copyright Ryan-Rhys Griffiths 2021
# Author: Ryan-Rhys Griffiths
"""
Scripts for benchmarking MLHGP-based Bayesian optimisation on synthetic functions with and without heteroscedastic noise.
"""

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

from acquisition_functions import heteroscedastic_expected_improvement, heteroscedastic_propose_location, \
    my_propose_location, my_expected_improvement, augmented_expected_improvement, heteroscedastic_augmented_expected_improvement
from BayesOpt.objective_funcs.synthetic_functions import hosaki_function, branin_function, goldstein_price_function


if __name__ == '__main__':

    exp_type = 'hetero'  # One of ['hetero', 'homoscedastic', 'noiseless']

    fill = True  # Whether to plot errorbars as fill or not.
    plot_collected = True  # Whether to plot collected data points on last random trial.
    penalty = 1  # penalty for aleatoric noise
    aleatoric_weight = 1
    noise_level = 0  # homoscedastic noise level. Should be 0 when heteroscedastic is True.
    if noise_level != 0:
        assert exp_type == 'homoscedastic'
    heteroscedastic = True
    if heteroscedastic:
        assert noise_level == 0
        assert exp_type == 'hetero'
    if heteroscedastic is not True and noise_level == 0:
        assert exp_type == 'noiseless'
    n_restarts = 20
    opt_func = 'goldstein'  # One of ['hosaki', 'branin', 'goldstein']
    grid_size = 10

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

        numpy_seed = i
        np.random.seed(numpy_seed)  # This seed changes the initialisation

        if opt_func == 'hosaki':
            bounds = np.array([[0.0, 5.0], [0.0, 5.0]])  # bounds of the Bayesian Optimisation problem for Hosaki
        else:
            bounds = np.array([[0.0, 1.0], [0.0, 1.0]])  # bounds for other test functions

        #  Initial noisy data points sampled uniformly at random from the input space.

        if opt_func == 'hosaki':
            X_init = np.random.uniform(0.0, 5.0, size=(grid_size**2, 2))
            Y_init = hosaki_function(X_init[:, 0], X_init[:, 1], heteroscedastic=heteroscedastic)
            x1_star = np.arange(0.0, 5.0, 0.5)
            x2_star = np.arange(0.0, 5.0, 0.5)
        else:
            X_init = np.random.uniform(0.0, 1.0, size=(grid_size**2, 2))
            x1_star = np.arange(0.0, 1.0, 0.1)
            x2_star = np.arange(0.0, 1.0, 0.1)
            if opt_func == 'branin':
                Y_init = branin_function(X_init[:, 0], X_init[:, 1], heteroscedastic=heteroscedastic)
            else:
                Y_init = goldstein_price_function(X_init[:, 0], X_init[:, 1], heteroscedastic=heteroscedastic)

        plot_sample = np.array(np.meshgrid(x1_star, x2_star)).T.reshape(-1, 2)  # Where 2 gives the dimensionality

        # Initialize samples
        if heteroscedastic:  # if heteroscedastic extract only the noisy evaluation of the objective function
            Y_init = Y_init[0]

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
        noise = 1.0  #  optimise noise for homoscedastic GP
        l_noise_init = 1.0
        sigma_f_noise_init = 1.0
        gp2_noise = 1.0
        num_iters = 10
        sample_size = 100

        rand_best_so_far = -300
        homo_best_so_far = -300  # value to beat
        het_best_so_far = -300
        aug_best_so_far = -300
        aug_het_best_so_far = -300
        rand_noise_best_so_far = 300  # value to beat
        homo_noise_best_so_far = 300
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
        rand_collected_x1 = []
        rand_collected_x2 = []
        homo_collected_x1 = []
        homo_collected_x2 = []
        het_collected_x1 = []
        het_collected_x2 = []
        aug_collected_x1 = []
        aug_collected_x2 = []
        aug_het_collected_x1 = []
        aug_het_collected_x2 = []

        for j in range(bayes_opt_iters):

            print(j)

            # random sampling baseline

            seed = bayes_opt_iters*i + j  # This seed changes the randomly sampled points (will be independent of initialisation)
            print(f'Seed is: {seed}')
            np.random.seed(seed)

            if opt_func == 'hosaki':
                random_x1_next = np.random.uniform(0.0, 5.0, size=(1,))
                random_x2_next = np.random.uniform(0.0, 5.0, size=(1,))
            else:
                random_x1_next = np.random.uniform(0.0, 1.0, size=(1,))
                random_x2_next = np.random.uniform(0.0, 1.0, size=(1,))

            random_X_next = np.array(np.meshgrid(random_x1_next, random_x2_next)).T.reshape(-1, 2)

            rand_collected_x1.append(random_x1_next)
            rand_collected_x2.append(random_x2_next)

            f_plot = True
            if j > 0:
                f_plot = False

            if opt_func == 'hosaki':
                random_Y_next = hosaki_function(random_x1_next, random_x2_next, noise=noise_level, heteroscedastic=heteroscedastic)
                if heteroscedastic:
                    random_Y_next = random_Y_next[0]
                    _, rand_noise_val, random_composite_obj_val = hosaki_function(random_x1_next, random_x2_next,
                                                                               noise=0.0, heteroscedastic=heteroscedastic)
                    random_composite_obj_val -= penalty*rand_noise_val
                else:
                    random_composite_obj_val = hosaki_function(random_x1_next, random_x2_next, noise=0.0, heteroscedastic=heteroscedastic)
            elif opt_func == 'branin':
                random_Y_next = branin_function(random_x1_next, random_x2_next, noise=noise_level, heteroscedastic=heteroscedastic)
                if heteroscedastic:
                    random_Y_next = random_Y_next[0]
                    _, rand_noise_val, random_composite_obj_val = branin_function(random_x1_next, random_x2_next,
                                                               noise=0.0, heteroscedastic=heteroscedastic)
                    random_composite_obj_val -= penalty*rand_noise_val
                else:
                    random_composite_obj_val = branin_function(random_x1_next, random_x2_next, noise=0.0, heteroscedastic=heteroscedastic)
            else:
                random_Y_next = goldstein_price_function(random_x1_next, random_x2_next, noise=noise_level, heteroscedastic=heteroscedastic)
                if heteroscedastic:
                    random_Y_next = random_Y_next[0]
                    _, rand_noise_val, random_composite_obj_val = goldstein_price_function(random_x1_next, random_x2_next,
                                                                      noise=0.0, heteroscedastic=heteroscedastic)
                    random_composite_obj_val -= penalty*rand_noise_val
                else:
                    random_composite_obj_val = goldstein_price_function(random_x1_next, random_x2_next, noise=0.0, heteroscedastic=heteroscedastic)

            f_plot = False

            if random_composite_obj_val > rand_best_so_far:
                rand_best_so_far = random_composite_obj_val
                rand_obj_val_list.append(random_composite_obj_val)
            else:
                rand_obj_val_list.append(rand_best_so_far)

            if heteroscedastic:
                if rand_noise_val < rand_noise_best_so_far:
                    rand_noise_best_so_far = rand_noise_val
                    rand_noise_val_list.append(rand_noise_val)
                else:
                    rand_noise_val_list.append(rand_noise_best_so_far)

            # Obtain next sampling point from the acquisition function (expected_improvement)

            homo_X_next = my_propose_location(my_expected_improvement, homo_X_sample, homo_Y_sample, noise, l_init, sigma_f_init,
                                              bounds, plot_sample, n_restarts=n_restarts, min_val=300)

            homo_collected_x1.append(homo_X_next[:, 0])
            homo_collected_x2.append(homo_X_next[:, 1])

            print(homo_X_next)

            # Obtain next noisy sample from the objective function
            if opt_func == 'hosaki':
                homo_Y_next = hosaki_function(homo_X_next[:, 0], homo_X_next[:, 1], noise=noise_level, heteroscedastic=heteroscedastic)
                if heteroscedastic:
                    homo_Y_next = homo_Y_next[0]
                    _, homo_noise_val, homo_composite_obj_val = hosaki_function(homo_X_next[:, 0], homo_X_next[:, 1],
                                                             noise=0.0, heteroscedastic=heteroscedastic)
                    homo_composite_obj_val -= penalty*homo_noise_val
                else:
                    homo_composite_obj_val = hosaki_function(homo_X_next[:, 0], homo_X_next[:, 1], noise=0.0, heteroscedastic=heteroscedastic)
            elif opt_func == 'branin':
                homo_Y_next = branin_function(homo_X_next[:, 0], homo_X_next[:, 1], noise=noise_level, heteroscedastic=heteroscedastic)
                if heteroscedastic:
                    homo_Y_next = homo_Y_next[0]
                    _, homo_noise_val, homo_composite_obj_val = branin_function(homo_X_next[:, 0], homo_X_next[:, 1],
                                                             noise=0.0, heteroscedastic=heteroscedastic)
                    homo_composite_obj_val -= penalty*homo_noise_val
                else:
                    homo_composite_obj_val = branin_function(homo_X_next[:, 0], homo_X_next[:, 1], noise=0.0, heteroscedastic=heteroscedastic)
            else:
                homo_Y_next = goldstein_price_function(homo_X_next[:, 0], homo_X_next[:, 1], noise=noise_level, heteroscedastic=heteroscedastic)
                if heteroscedastic:
                    homo_Y_next = homo_Y_next[0]
                    _, homo_noise_val, homo_composite_obj_val = goldstein_price_function(homo_X_next[:, 0], homo_X_next[:, 1],
                                                                    noise=0.0, heteroscedastic=heteroscedastic)
                    homo_composite_obj_val -= penalty*homo_noise_val
                else:
                    homo_composite_obj_val = goldstein_price_function(homo_X_next[:, 0], homo_X_next[:, 1], noise=0.0, heteroscedastic=heteroscedastic)

            if homo_composite_obj_val > homo_best_so_far:
                homo_best_so_far = homo_composite_obj_val
                homo_obj_val_list.append(homo_composite_obj_val)
            else:
                homo_obj_val_list.append(homo_best_so_far)

            if heteroscedastic:
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
                                                          plot_sample, n_restarts=n_restarts, min_val=300, aleatoric_weight=aleatoric_weight)

            het_collected_x1.append(het_X_next[:, 0])
            het_collected_x2.append(het_X_next[:, 1])

            # Obtain next noisy sample from the objective function
            if opt_func == 'hosaki':
                het_Y_next = hosaki_function(het_X_next[:, 0], het_X_next[:, 1], noise=noise_level, heteroscedastic=heteroscedastic)
                if heteroscedastic:
                    het_Y_next = het_Y_next[0]
                    _, het_noise_val, het_composite_obj_val = hosaki_function(het_X_next[:, 0], het_X_next[:, 1],
                                                            noise=0.0, heteroscedastic=heteroscedastic)
                    het_composite_obj_val -= penalty*het_noise_val
                else:
                    het_composite_obj_val = hosaki_function(het_X_next[:, 0], het_X_next[:, 1], noise=0.0, heteroscedastic=heteroscedastic)
            elif opt_func == 'branin':
                het_Y_next = branin_function(het_X_next[:, 0], het_X_next[:, 1], noise=noise_level, heteroscedastic=heteroscedastic)
                if heteroscedastic:
                    het_Y_next = het_Y_next[0]
                    _, het_noise_val, het_composite_obj_val = branin_function(het_X_next[:, 0], het_X_next[:, 1],
                                                            noise=0.0, heteroscedastic=heteroscedastic)
                    het_composite_obj_val -= penalty*het_noise_val
                else:
                    het_composite_obj_val = branin_function(het_X_next[:, 0], het_X_next[:, 1], noise=0.0, heteroscedastic=heteroscedastic)
            else:
                het_Y_next = goldstein_price_function(het_X_next[:, 0], het_X_next[:, 1], noise=noise_level, heteroscedastic=heteroscedastic)
                if heteroscedastic:
                    het_Y_next = het_Y_next[0]
                    _, het_noise_val, het_composite_obj_val = goldstein_price_function(het_X_next[:, 0], het_X_next[:, 1],
                                                                   noise=0.0, heteroscedastic=heteroscedastic)
                    het_composite_obj_val -= penalty*het_noise_val
                else:
                    het_composite_obj_val = goldstein_price_function(het_X_next[:, 0], het_X_next[:, 1], noise=0.0, heteroscedastic=heteroscedastic)

            if het_composite_obj_val > het_best_so_far:
                het_best_so_far = het_composite_obj_val
                het_obj_val_list.append(het_composite_obj_val)
            else:
                het_obj_val_list.append(het_best_so_far)

            if heteroscedastic:
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
                                             bounds, plot_sample, n_restarts=n_restarts, min_val=300, aleatoric_weight=aleatoric_weight, aei=True)

            aug_collected_x1.append(aug_X_next[:, 0])
            aug_collected_x2.append(aug_X_next[:, 1])

            # Obtain next noisy sample from the objective function
            if opt_func == 'hosaki':
                aug_Y_next = hosaki_function(aug_X_next[:, 0], aug_X_next[:, 1], noise=noise_level, heteroscedastic=heteroscedastic)
                if heteroscedastic:
                    aug_Y_next = aug_Y_next[0]
                    _, aug_noise_val, aug_composite_obj_val = hosaki_function(aug_X_next[:, 0], aug_X_next[:, 1],
                                                            noise=0.0, heteroscedastic=heteroscedastic)
                    aug_composite_obj_val -= penalty*aug_noise_val
                else:
                    aug_composite_obj_val = hosaki_function(aug_X_next[:, 0], aug_X_next[:, 1], noise=0.0, heteroscedastic=heteroscedastic)
            elif opt_func == 'branin':
                aug_Y_next = branin_function(aug_X_next[:, 0], aug_X_next[:, 1], noise=noise_level, heteroscedastic=heteroscedastic)
                if heteroscedastic:
                    aug_Y_next = aug_Y_next[0]
                    _, aug_noise_val, aug_composite_obj_val = branin_function(aug_X_next[:, 0], aug_X_next[:, 1],
                                                            noise=0.0, heteroscedastic=heteroscedastic)
                    aug_composite_obj_val -= penalty*aug_noise_val
                else:
                    aug_composite_obj_val = branin_function(aug_X_next[:, 0], aug_X_next[:, 1], noise=0.0, heteroscedastic=heteroscedastic)
            else:
                aug_Y_next = goldstein_price_function(aug_X_next[:, 0], aug_X_next[:, 1], noise=noise_level, heteroscedastic=heteroscedastic)
                if heteroscedastic:
                    aug_Y_next = aug_Y_next[0]
                    _, aug_noise_val, aug_composite_obj_val = goldstein_price_function(aug_X_next[:, 0], aug_X_next[:, 1],
                                                                   noise=0.0, heteroscedastic=heteroscedastic)
                    aug_composite_obj_val -= penalty*aug_noise_val
                else:
                    aug_composite_obj_val = goldstein_price_function(aug_X_next[:, 0], aug_X_next[:, 1], noise=0.0, heteroscedastic=heteroscedastic)

            if aug_composite_obj_val > aug_best_so_far:
                aug_best_so_far = aug_composite_obj_val
                aug_obj_val_list.append(aug_composite_obj_val)
            else:
                aug_obj_val_list.append(aug_best_so_far)

            if heteroscedastic:
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
                                                          plot_sample, n_restarts=n_restarts, min_val=300, aleatoric_weight=500)

            aug_het_collected_x1.append(aug_het_X_next[:, 0])
            aug_het_collected_x2.append(aug_het_X_next[:, 1])

            # Obtain next noisy sample from the objective function
            if opt_func == 'hosaki':
                aug_het_Y_next = hosaki_function(aug_het_X_next[:, 0], aug_het_X_next[:, 1], noise=noise_level, heteroscedastic=heteroscedastic)
                if heteroscedastic:
                    aug_het_Y_next = aug_het_Y_next[0]
                    _, aug_het_noise_val, aug_het_composite_obj_val = hosaki_function(aug_het_X_next[:, 0], aug_het_X_next[:, 1],
                                                                noise=0.0, heteroscedastic=heteroscedastic)
                    aug_het_composite_obj_val -= penalty*aug_het_noise_val
                else:
                    aug_het_composite_obj_val = hosaki_function(aug_het_X_next[:, 0], aug_het_X_next[:, 1], noise=0.0, heteroscedastic=heteroscedastic)
            elif opt_func == 'branin':
                aug_het_Y_next = branin_function(aug_het_X_next[:, 0], aug_het_X_next[:, 1], noise=noise_level, heteroscedastic=heteroscedastic)
                if heteroscedastic:
                    aug_het_Y_next = aug_het_Y_next[0]
                    _, aug_het_noise_val, aug_het_composite_obj_val = branin_function(aug_het_X_next[:, 0], aug_het_X_next[:, 1],
                                                                noise=0.0, heteroscedastic=heteroscedastic)
                    aug_het_composite_obj_val -= penalty*aug_het_noise_val
                else:
                    aug_het_composite_obj_val = branin_function(aug_het_X_next[:, 0], aug_het_X_next[:, 1], noise=0.0, heteroscedastic=heteroscedastic)
            else:
                aug_het_Y_next = goldstein_price_function(aug_het_X_next[:, 0], aug_het_X_next[:, 1], noise=noise_level, heteroscedastic=heteroscedastic)
                if heteroscedastic:
                    aug_het_Y_next = aug_het_Y_next[0]
                    _, aug_het_noise_val, aug_het_composite_obj_val = goldstein_price_function(aug_het_X_next[:, 0], aug_het_X_next[:, 1],
                                                                                          noise=0.0, heteroscedastic=heteroscedastic)
                    aug_het_composite_obj_val -= penalty*aug_het_noise_val
                else:
                    aug_het_composite_obj_val = goldstein_price_function(aug_het_X_next[:, 0], aug_het_X_next[:, 1], noise=0.0, heteroscedastic=heteroscedastic)

            if aug_het_composite_obj_val > aug_het_best_so_far:
                aug_het_best_so_far = aug_het_composite_obj_val
                aug_het_obj_val_list.append(aug_het_composite_obj_val)
            else:
                aug_het_obj_val_list.append(aug_het_best_so_far)

            if heteroscedastic:
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

        if heteroscedastic:
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

    # results are negated to turn problem into minimisation for consistency.
    rand_means = -rand_running_sum / random_trials
    rand_errs = (np.sqrt(rand_squares / random_trials - rand_means **2, dtype=np.float64))/np.sqrt(random_trials)
    homo_means = -homo_running_sum / random_trials
    homo_errs = (np.sqrt(homo_squares / random_trials - homo_means ** 2, dtype=np.float64))/np.sqrt(random_trials)
    hetero_means = -hetero_running_sum / random_trials
    hetero_errs = (np.sqrt(hetero_squares / random_trials - hetero_means ** 2, dtype=np.float64))/np.sqrt(random_trials)
    aug_means = -aug_running_sum / random_trials
    aug_errs = (np.sqrt(aug_squares / random_trials - aug_means ** 2, dtype=np.float64))/np.sqrt(random_trials)
    aug_het_means = -aug_het_running_sum / random_trials
    aug_het_errs = (np.sqrt(aug_het_squares / random_trials - aug_het_means **2, dtype=np.float64))/np.sqrt(random_trials)

    if heteroscedastic:
        rand_noise_means = rand_noise_running_sum / random_trials
        homo_noise_means = homo_noise_running_sum / random_trials
        hetero_noise_means = hetero_noise_running_sum / random_trials
        aug_noise_means = aug_noise_running_sum / random_trials
        aug_het_noise_means = aug_het_noise_running_sum / random_trials
        rand_noise_errs = (np.sqrt(rand_noise_squares / random_trials - rand_noise_means ** 2))/np.sqrt(random_trials)
        homo_noise_errs = (np.sqrt(homo_noise_squares / random_trials - homo_noise_means ** 2))/np.sqrt(random_trials)
        hetero_noise_errs = (np.sqrt(hetero_noise_squares / random_trials - hetero_noise_means ** 2))/np.sqrt(random_trials)
        aug_noise_errs = (np.sqrt(aug_noise_squares / random_trials - aug_noise_means ** 2))/np.sqrt(random_trials)
        aug_het_noise_errs = (np.sqrt(aug_het_noise_squares / random_trials - aug_het_noise_means ** 2))/np.sqrt(random_trials)

    print('List of average random values is: ' + str(rand_means))
    print('List of random errors is: ' + str(rand_errs))
    print('List of average homoscedastic values is: ' + str(homo_means))
    print('List of homoscedastic errors is: ' + str(homo_errs))
    print('List of average heteroscedastic values is ' + str(hetero_means))
    print('List of heteroscedastic errors is: ' + str(hetero_errs))
    print('List of average AEI values is: ' + str(aug_means))
    print('List of AEI errors is: ' + str(aug_errs))
    print('List of average het-AEI values is: ' + str(aug_het_means))
    print('List of het-AEI errors is: ' + str(aug_het_errs))

    iter_x = np.arange(1, bayes_opt_iters + 1)

    # clear figure from previous fplot
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

    if fill:
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
    else:
        plt.errorbar(iter_x, homo_means, yerr=np.concatenate((homo_means - lower_homo, upper_homo - homo_means)).reshape((2,5)), color='r', label='Homoscedastic', capsize=5)
        plt.errorbar(iter_x, hetero_means, yerr=np.concatenate((hetero_means - lower_hetero, upper_hetero - hetero_means)).reshape((2,5)), color='b', label='Heteroscedastic ANPEI', capsize=5)
        plt.errorbar(iter_x, rand_means, yerr=np.concatenate((rand_means - lower_rand, upper_rand - rand_means)).reshape((2,5)), color='g', label='Random Sampling', capsize=5)
        plt.errorbar(iter_x, aug_means, yerr=np.concatenate((aug_means - lower_aei, upper_aei - aug_means)).reshape((2,5)), color='c', label='Homoscedastic AEI', capsize=5)
        plt.errorbar(iter_x, aug_het_means, yerr=np.concatenate((aug_het_means - lower_het_aei, upper_het_aei - aug_het_means)).reshape((2,5)), color='m', label='Heteroscedastic AEI', capsize=5)

    plt.title('Best Objective Function Value Found so Far', fontsize=16)
    plt.xlabel('Function Evaluations', fontsize=14)
    if heteroscedastic:
        if penalty != 1:
            plt.ylabel('f(x) +' + str(penalty) + 'g(x)', fontsize=14)
        else:
            plt.ylabel('f(x) + g(x)', fontsize=14)
    else:
        plt.ylabel('f(x)', fontsize=14)
    plt.tick_params(labelsize=14)
    plt.legend(loc='lower left', bbox_to_anchor=(0.0, -0.425), ncol=3, borderaxespad=0, fontsize=14, frameon=False)
    if noise_level > 0:
        tag = 'with_noise_' + str(noise_level)
    else:
        if heteroscedastic:
            tag = 'heteroscedastic'
        else:
            tag = ''
    plt.savefig('new_figures/{}/{}_{}_iters_{}_random_trials_and_grid_size_of_{}_and_seed_{}'
                '_hundred_times_penalty_is_{}_aleatoric_weight_is_{}_{}_'.
                format(opt_func, exp_type, bayes_opt_iters, random_trials, grid_size, numpy_seed, int(100*penalty), aleatoric_weight, tag), bbox_inches='tight')

    plt.close()

    # clear figure from previous fplot returns if fiddling with form of function
    plt.cla()

    if heteroscedastic:
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

        #best_noise_plot = np.zeros(len(iter_x))

        if fill:
            #plt.plot(iter_x, best_noise_plot, '--', color='k', label='Optimal')
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
        else:
            plt.errorbar(iter_x, homo_noise_means, yerr=np.concatenate((homo_noise_means - lower_noise_homo, upper_noise_homo - homo_noise_means)).reshape((2,5)), color='r', label='Homoscedastic', capsize=5)
            plt.errorbar(iter_x, hetero_noise_means, yerr=np.concatenate((hetero_noise_means - lower_noise_hetero, upper_noise_hetero - hetero_noise_means)).reshape((2,5)), color='b', label='Heteroscedastic ANPEI', capsize=5)
            plt.errorbar(iter_x, rand_noise_means, yerr=np.concatenate((rand_noise_means - lower_noise_rand, upper_noise_rand - rand_noise_means)).reshape((2,5)), color='g', label='Random Sampling', capsize=5)
            plt.errorbar(iter_x, aug_noise_means, yerr=np.concatenate((aug_noise_means - lower_noise_aei, upper_noise_aei - aug_noise_means)).reshape((2,5)), color='c', label='Homoscedastic AEI', capsize=5)
            plt.errorbar(iter_x, aug_het_noise_means, yerr=np.concatenate((aug_het_noise_means - lower_noise_het_aei, upper_noise_het_aei - aug_het_noise_means)).reshape((2,5)), color='m', label='Heteroscedastic AEI', capsize=5)

        plt.title('Lowest Aleatoric Noise Found so Far', fontsize=16)
        plt.xlabel('Function Evaluations', fontsize=14)
        plt.ylabel('g(x)', fontsize=14)
        plt.tick_params(labelsize=14)
        plt.legend(loc='lower left', bbox_to_anchor=(0.0, -0.425), ncol=3, borderaxespad=0, fontsize=14, frameon=False)
        plt.savefig('new_figures/{}/heteroscedastic_bayesopt_plot{}_iters_{}_random_trials_and_grid_size_of_{}_and_seed_{}_'
                    'noise_only_hundred_times_penalty_is_{}_aleatoric_weight_is_{}_new_ae'.
                    format(opt_func, bayes_opt_iters, random_trials, grid_size, numpy_seed, int(100*penalty), aleatoric_weight), bbox_inches='tight')

    if plot_collected:

        plt.cla()

        plt.plot(np.array(rand_collected_x1), np.array(rand_collected_x2), '+', color='tab:orange', markersize='12', linewidth='8')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.xlim(bounds[0][0], bounds[0][1])
        plt.ylim(bounds[1][0], bounds[1][1])
        plt.title('Collected Data Points')
        plt.savefig('new_figures/{}/collected_points/bayesopt_plot{}_iters_{}_random_trials_and'
                '_grid_size_of_{}_and_seed_{}_with_het_aei_full_unc_new_rand'.format(opt_func, bayes_opt_iters, random_trials, grid_size, numpy_seed))
        plt.close()
        plt.cla()

        plt.plot(np.array(homo_collected_x1), np.array(homo_collected_x2), '+', color='tab:blue', markersize='12', linewidth='8')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.xlim(bounds[0][0], bounds[0][1])
        plt.ylim(bounds[1][0], bounds[1][1])
        plt.title('Collected Data Points')
        plt.savefig('new_figures/{}/collected_points/bayesopt_plot{}_iters_{}_random_trials_and'
                '_grid_size_of_{}_and_seed_{}_with_het_aei_full_unc_new_rand_homo'.format(opt_func, bayes_opt_iters, random_trials, grid_size, numpy_seed))
        plt.close()
        plt.cla()

        plt.plot(np.array(het_collected_x1), np.array(het_collected_x2), '+', color='tab:green', markersize='12', linewidth='8')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.xlim(bounds[0][0], bounds[0][1])
        plt.ylim(bounds[1][0], bounds[1][1])
        plt.title('Collected Data Points')
        plt.savefig('new_figures/{}/collected_points/bayesopt_plot{}_iters_{}_random_trials_and'
                '_grid_size_of_{}_and_seed_{}_with_het_aei_full_unc_new_het_anpei'.format(opt_func, bayes_opt_iters, random_trials, grid_size, numpy_seed))
        plt.close()
        plt.cla()

        plt.plot(np.array(aug_collected_x1), np.array(aug_collected_x2), '+', color='tab:red', markersize='12', linewidth='8')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.xlim(bounds[0][0], bounds[0][1])
        plt.ylim(bounds[1][0], bounds[1][1])
        plt.title('Collected Data Points')
        plt.savefig('new_figures/{}/collected_points/bayesopt_plot{}_iters_{}_random_trials_and'
                '_grid_size_of_{}_and_seed_{}_with_het_aei_full_unc_new_rand_aug'.format(opt_func, bayes_opt_iters, random_trials, grid_size, numpy_seed))
        plt.close()
        plt.cla()

        plt.plot(np.array(aug_het_collected_x1), np.array(aug_het_collected_x2), '+', color='tab:purple', markersize='12', linewidth='8')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.xlim(bounds[0][0], bounds[0][1])
        plt.ylim(bounds[1][0], bounds[1][1])
        plt.title('Collected Data Points')
        plt.savefig('new_figures/{}/collected_points/bayesopt_plot{}_iters_{}_random_trials_and'
                '_grid_size_of_{}_and_seed_{}_with_het_aei_full_unc_new_rand_aug_het'.format(opt_func, bayes_opt_iters, random_trials, grid_size, numpy_seed))
        plt.close()


    # Save data for cosmetic plotting

    np.savetxt(f'synth_saved_data/{exp_type}/{opt_func}/rand_means.txt', rand_means)
    np.savetxt(f'synth_saved_data/{exp_type}/{opt_func}/homo_means.txt', homo_means)
    np.savetxt(f'synth_saved_data/{exp_type}/{opt_func}/hetero_means.txt', hetero_means)
    np.savetxt(f'synth_saved_data/{exp_type}/{opt_func}/aug_means.txt', aug_means)
    np.savetxt(f'synth_saved_data/{exp_type}/{opt_func}/aug_het_means.txt', aug_het_means)

    np.savetxt(f'synth_saved_data/{exp_type}/{opt_func}/lower_rand.txt', lower_rand)
    np.savetxt(f'synth_saved_data/{exp_type}/{opt_func}/upper_rand.txt', upper_rand)
    np.savetxt(f'synth_saved_data/{exp_type}/{opt_func}/lower_homo.txt', lower_homo)
    np.savetxt(f'synth_saved_data/{exp_type}/{opt_func}/upper_homo.txt', upper_homo)
    np.savetxt(f'synth_saved_data/{exp_type}/{opt_func}/lower_hetero.txt', lower_hetero)
    np.savetxt(f'synth_saved_data/{exp_type}/{opt_func}/upper_hetero.txt', upper_hetero)
    np.savetxt(f'synth_saved_data/{exp_type}/{opt_func}/lower_aei.txt', lower_aei)
    np.savetxt(f'synth_saved_data/{exp_type}/{opt_func}/upper_aei.txt', upper_aei)
    np.savetxt(f'synth_saved_data/{exp_type}/{opt_func}/lower_het_aei.txt', lower_het_aei)
    np.savetxt(f'synth_saved_data/{exp_type}/{opt_func}/upper_het_aei.txt', upper_het_aei)

    if heteroscedastic:
        np.savetxt(f'synth_saved_data/{exp_type}/{opt_func}/rand_noise_means.txt', rand_noise_means)
        np.savetxt(f'synth_saved_data/{exp_type}/{opt_func}/homo_noise_means.txt', homo_noise_means)
        np.savetxt(f'synth_saved_data/{exp_type}/{opt_func}/hetero_noise_means.txt', hetero_noise_means)
        np.savetxt(f'synth_saved_data/{exp_type}/{opt_func}/aug_noise_means.txt', aug_noise_means)
        np.savetxt(f'synth_saved_data/{exp_type}/{opt_func}/aug_het_noise_means.txt', aug_het_noise_means)

        np.savetxt(f'synth_saved_data/{exp_type}/{opt_func}/lower_noise_rand.txt', lower_noise_rand)
        np.savetxt(f'synth_saved_data/{exp_type}/{opt_func}/upper_noise_rand.txt', upper_noise_rand)
        np.savetxt(f'synth_saved_data/{exp_type}/{opt_func}/lower_noise_homo.txt', lower_noise_homo)
        np.savetxt(f'synth_saved_data/{exp_type}/{opt_func}/upper_noise_homo.txt', upper_noise_homo)
        np.savetxt(f'synth_saved_data/{exp_type}/{opt_func}/lower_noise_hetero.txt', lower_noise_hetero)
        np.savetxt(f'synth_saved_data/{exp_type}/{opt_func}/upper_noise_hetero.txt', upper_noise_hetero)
        np.savetxt(f'synth_saved_data/{exp_type}/{opt_func}/lower_noise_aei.txt', lower_noise_aei)
        np.savetxt(f'synth_saved_data/{exp_type}/{opt_func}/upper_noise_aei.txt', upper_noise_aei)
        np.savetxt(f'synth_saved_data/{exp_type}/{opt_func}/lower_noise_het_aei.txt', lower_noise_het_aei)
        np.savetxt(f'synth_saved_data/{exp_type}/{opt_func}/upper_noise_het_aei.txt', upper_noise_het_aei)
