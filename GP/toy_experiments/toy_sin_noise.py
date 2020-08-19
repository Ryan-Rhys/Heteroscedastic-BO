# Copyright Lee Group 2019
# Author: Ryan-Rhys Griffiths (Modified by Alex Aldrick)
"""
This module contains the code for benchmarking heteroscedastic Bayesian Optimisation on a 1D toy problem.
"""

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from numpy.core._multiarray_umath import ndarray

from acquisition_functions import heteroscedastic_expected_improvement, heteroscedastic_propose_location, \
    my_propose_location, my_expected_improvement, augmented_expected_improvement, \
    heteroscedastic_augmented_expected_improvement
from objective_functions import linear_sin_noise, max_sin_noise_objective

if __name__ == '__main__':

    modification = True  # Switches between sin(x) - False and sin(x) + 0.05x - True
    coefficient = 0.2  # tunes the relative size of the maxima in the function (used when modification = True)

    # Number of iterations
    random_trials = 10
    bayes_opt_iters = 5
    random_sampling_iters = 5

    # We perform random trials of Bayesian Optimisation

    rand_running_sum = np.zeros(random_sampling_iters)
    rand_squares = np.zeros(random_sampling_iters)
    homo_running_sum = np.zeros(bayes_opt_iters)
    homo_squares = np.zeros(
        bayes_opt_iters)  # Following the single-pass estimator given on pg. 192 of mathematics for machine learning
    hetero_running_sum = np.zeros(bayes_opt_iters)
    hetero_squares = np.zeros(bayes_opt_iters)

    # We compute the objective corresponding to aleatoric noise only

    homo_noise_running_sum = np.zeros(bayes_opt_iters)
    homo_noise_squares = np.zeros(
        bayes_opt_iters)  # Following the single-pass estimator given on pg. 192 of mathematics for machine learning
    hetero_noise_running_sum = np.zeros(bayes_opt_iters)
    hetero_noise_squares = np.zeros(bayes_opt_iters)

    for i in range(random_trials): #random trials to get average across x number of trials

        numpy_seed = i + 62
        np.random.seed(numpy_seed)

        noise_coeff = 0.25  # noise coefficient will be noise(X) will be linear e.g. 0.2 * X
        bounds = np.array([0, 10]).reshape(-1, 1)  # bounds of the Bayesian Optimisation problem.

        #  Initial noisy data points sampled uniformly at random from the input space.

        init_num_samples = 3  # all un-named plots were 33 initial samples
        X_init = np.random.uniform(0, 10, init_num_samples).reshape(-1,
                                                                    1)  # sample 7 points at random from the bounds to initialise with
        plot_sample = np.linspace(0, 10, 50).reshape(-1, 1)  # samples for plotting purposes

        Y_init = linear_sin_noise(X_init, noise_coeff, plot_sample, coefficient, modification, fplot=True)

        # Initialize samples
        homo_X_sample = X_init.reshape(-1, 1)
        homo_Y_sample = Y_init.reshape(-1, 1)
        het_X_sample = X_init.reshape(-1, 1)
        het_Y_sample = Y_init.reshape(-1, 1)

        # initial GP hypers

        l_init = 1.0
        sigma_f_init = 1.0
        noise = 1.0  # need to be careful about how we set this because it's not currently being optimised in the code (see reviewer comment)
        l_noise_init = 1.0
        sigma_f_noise_init = 1.0
        gp2_noise = 1.0
        num_iters = 10
        sample_size = 100

        rand_best_so_far = - 300  # value to beat
        homo_best_so_far = -300
        het_best_so_far = -300
        rand_obj_val_list = []
        homo_obj_val_list = []
        het_obj_val_list = []
        homo_noise_val_list = []
        het_noise_val_list = []

        for j in range(random_sampling_iters):
            print('Random Sampling Iteration: ' + str(j))

            # take random point from uniform distribution
            rand_X_next = np.random.uniform(0, 10)  # this just takes X not the sin function itself
            # check if random point's Y value is better than best so far
            rand_composite_obj_val, rand_noise_val = max_sin_noise_objective(rand_X_next, noise_coeff, coefficient,
                                                                             modification, fplot=False)
            if rand_composite_obj_val > rand_best_so_far:
                rand_best_so_far = rand_composite_obj_val
                rand_obj_val_list.append(rand_composite_obj_val)
            else:
                rand_obj_val_list.append(rand_best_so_far)
            # if yes, save it, if no, save best so far into list of best y-value per iteration in rand_composite_obj_val

        rand_running_sum += np.array(rand_obj_val_list) # just the way to average out across all random trials
        rand_squares += np.array(rand_obj_val_list) ** 2 # likewise for errors

        for i in range(bayes_opt_iters):
        #number of BO iterations i.e. number of times sampled from black-box function using the acquisition function.

            print(i)

            # Obtain next sampling point from the acquisition function (expected_improvement)

            homo_X_next = my_propose_location(my_expected_improvement, homo_X_sample, homo_Y_sample, noise, l_init,
                                              sigma_f_init, bounds, plot_sample, n_restarts=3, min_val=300)

            # Obtain next noisy sample from the objective function
            homo_Y_next = linear_sin_noise(homo_X_next, noise_coeff, plot_sample, coefficient, modification, fplot=False)
            homo_composite_obj_val, homo_noise_val = max_sin_noise_objective(homo_X_next, noise_coeff, coefficient,
                                                                             modification, fplot=False)
            # if the new Y-value is better than our best so far, save it as best so far and append it to best Y-values list in *_composite_obj_val
            if homo_composite_obj_val > homo_best_so_far:
                homo_best_so_far = homo_composite_obj_val
                homo_obj_val_list.append(homo_composite_obj_val)
            else:
                homo_obj_val_list.append(homo_best_so_far)

            # Add sample to previous samples
            homo_X_sample = np.vstack((homo_X_sample, homo_X_next))
            homo_Y_sample = np.vstack((homo_Y_sample, homo_Y_next))

            # Obtain next sampling point from the het acquisition function (ANPEI)

            het_X_next = heteroscedastic_propose_location(heteroscedastic_expected_improvement, het_X_sample,
                                                          het_Y_sample, noise, l_init, sigma_f_init, l_noise_init,
                                                          sigma_f_noise_init, gp2_noise, num_iters, sample_size, bounds,
                                                          plot_sample, n_restarts=3, min_val=300)

            # Obtain next noisy sample from the objective function
            het_Y_next = linear_sin_noise(het_X_next, noise_coeff, plot_sample, coefficient, modification, fplot=False)
            het_composite_obj_val, het_noise_val = max_sin_noise_objective(het_X_next, noise_coeff, coefficient,
                                                                           modification, fplot=False)

            if het_composite_obj_val > het_best_so_far:
                het_best_so_far = het_composite_obj_val
                het_obj_val_list.append(het_composite_obj_val)
            else:
                het_obj_val_list.append(het_best_so_far)

            # Add sample to previous samples
            het_X_sample = np.vstack((het_X_sample, het_X_next))
            het_Y_sample = np.vstack((het_Y_sample, het_Y_next))

        homo_running_sum += np.array(homo_obj_val_list) # adding the best values in order of iteration on top of each other element wise
        homo_squares += np.array(homo_obj_val_list) ** 2
        hetero_running_sum += np.array(het_obj_val_list)
        hetero_squares += np.array(het_obj_val_list) ** 2

    rand_means = rand_running_sum / random_trials
    homo_means = homo_running_sum / random_trials
    hetero_means = hetero_running_sum / random_trials
    rand_errs = np.sqrt(rand_squares / random_trials - rand_means ** 2)
    homo_errs = np.sqrt(homo_squares / random_trials - homo_means ** 2)
    hetero_errs = np.sqrt(hetero_squares / random_trials - hetero_means ** 2)

    print('List of average random sampling values is: ' + str(rand_means))
    print('List of random sampling errors is:' + str(rand_errs))
    print('List of average homoscedastic values is: ' + str(homo_means))
    print('List of homoscedastic errors is: ' + str(homo_errs))
    print('List of average heteroscedastic values is ' + str(hetero_means))
    print('List of heteroscedastic errors is: ' + str(hetero_errs))

    iter_x = np.arange(1, bayes_opt_iters + 1)
    iter_rand_x = np.arange(1, random_sampling_iters + 1)

    # clear figure from previous fplot returns if fiddling with form of function
    plt.cla()

    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.plot(iter_x, homo_means, color='r', label='Homoscedastic')
    plt.plot(iter_x, hetero_means, color='b', label='Heteroscedastic ANPEI')
    plt.plot(iter_rand_x, rand_means, color='g', label='Random Sampling')
    lower_rand = np.array(rand_means) - np.array(rand_errs)
    upper_rand = np.array(rand_means) + np.array(rand_errs)
    lower_homo = np.array(homo_means) - np.array(homo_errs)
    upper_homo = np.array(homo_means) + np.array(homo_errs)
    lower_hetero = np.array(hetero_means) - np.array(hetero_errs)
    upper_hetero = np.array(hetero_means) + np.array(hetero_errs)
    plt.fill_between(iter_x, lower_homo, upper_homo, color='r', label='Homoscedastic', alpha=0.1)
    plt.fill_between(iter_x, lower_hetero, upper_hetero, color='b', label='Heteroscedastic ANPEI', alpha=0.1)
    plt.fill_between(iter_rand_x, lower_rand, upper_rand, color='g', label='Random Sampling', alpha=0.1)
    plt.title('Best Objective Function Value Found so Far')
    plt.xlabel('Number of Function Evaluations')
    plt.ylabel('Objective Function Value - Noise')
    plt.legend(loc=4)
    plt.savefig('toy_figures/bayesopt_plot{}_iters_{}_random_trials_and_{}_coefficient_times_100_and_noise_coeff_times_'
                '100_of_{}_init_num_samples_of_{}_and_seed_{}'.format(bayes_opt_iters, random_trials,
                                                                      int(coefficient * 100), int(noise_coeff * 100),
                                                                      init_num_samples, numpy_seed))
