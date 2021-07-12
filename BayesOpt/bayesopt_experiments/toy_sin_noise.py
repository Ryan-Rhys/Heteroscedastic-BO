# Copyright Lee Group 2019
# Author: Ryan-Rhys Griffiths (Modified by Alex Aldrick)
"""
This module contains the code for benchmarking heteroscedastic Bayesian Optimisation on a 1D toy problem.
"""

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

from acquisition_functions import heteroscedastic_expected_improvement, heteroscedastic_propose_location, \
    my_propose_location, my_expected_improvement, augmented_expected_improvement, \
    heteroscedastic_augmented_expected_improvement
from objective_functions import linear_sin_noise, max_sin_noise_objective

if __name__ == '__main__':

    fill = True  # Whether to fill errorbars or not
    coefficient = 0.2  # tunes the relative size of the maxima in the function (used when modification = True)
    penalty = 1
    aleatoric_penalty = 1
    noise_coeff = 0.5  # noise coefficient will be noise(X) will be linear e.g. 0.2 * X

    # Number of iterations
    random_trials = 49
    bayes_opt_iters = 5

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

    fplot = True  # plot data

    for i in range(random_trials):  # random trials to get average across x number of trials

        numpy_seed = i
        np.random.seed(numpy_seed)

        bounds = np.array([0, 10]).reshape(-1, 1)  # bounds of the Bayesian Optimisation problem.

        #  Initial noisy data points sampled uniformly at random from the input space.

        init_num_samples = 25
        X_init = np.random.uniform(0, 10, init_num_samples).reshape(-1, 1)  # sample 10 points at random from the bounds to initialise with
        plot_sample = np.linspace(0, 10, 50).reshape(-1, 1)  # samples for plotting purposes

        if i > 0:
            fplot = False

        Y_init = linear_sin_noise(X_init, noise_coeff, plot_sample, coefficient, fplot=fplot)

        # Initialize samples
        homo_X_sample = X_init.reshape(-1, 1)
        homo_Y_sample = Y_init.reshape(-1, 1)
        het_X_sample = X_init.reshape(-1, 1)
        het_Y_sample = Y_init.reshape(-1, 1)
        aug_X_sample = X_init.reshape(-1, 1)
        aug_Y_sample = Y_init.reshape(-1, 1)
        aug_het_X_sample = X_init.reshape(-1, 1)
        aug_het_Y_sample = Y_init.reshape(-1, 1)

        # initial BayesOpt hypers

        l_init = 1.0
        sigma_f_init = 1.0
        noise = 1.0
        l_noise_init = 1.0
        sigma_f_noise_init = 1.0
        gp2_noise = 1.0
        num_iters = 10
        sample_size = 100

        rand_best_so_far = -300  # value to beat
        homo_best_so_far = -300
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

        for i in range(bayes_opt_iters):

            # number of BO iterations i.e. number of times sampled from black-box function using the acquisition function.
            # random sampling first
            # take random point from uniform distribution
            rand_X_next = np.random.uniform(0, 10)  # this just takes X not the sin function itself
            # check if random point's Y value is better than best so far
            rand_composite_obj_val, rand_noise_val = max_sin_noise_objective(rand_X_next, noise_coeff, coefficient,
                                                                             fplot=False, penalty=penalty)
            if rand_composite_obj_val > rand_best_so_far:
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

            print(i)

            # Obtain next sampling point from the acquisition function (expected_improvement)

            homo_X_next = my_propose_location(my_expected_improvement, homo_X_sample, homo_Y_sample, noise, l_init,
                                              sigma_f_init, bounds, plot_sample, n_restarts=3, min_val=300)

            # Obtain next noisy sample from the objective function
            homo_Y_next = linear_sin_noise(homo_X_next, noise_coeff, plot_sample, coefficient, fplot=False)
            homo_composite_obj_val, homo_noise_val = max_sin_noise_objective(homo_X_next, noise_coeff, coefficient, fplot=False, penalty=penalty)

            # if the new Y-value is better than our best so far, save it as best so far and append it to best Y-values list in *_composite_obj_val

            if homo_composite_obj_val > homo_best_so_far:
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
                                                          plot_sample, n_restarts=3, min_val=300, aleatoric_weight=aleatoric_penalty)

            # Obtain next noisy sample from the objective function
            het_Y_next = linear_sin_noise(het_X_next, noise_coeff, plot_sample, coefficient, fplot=False)
            het_composite_obj_val, het_noise_val = max_sin_noise_objective(het_X_next, noise_coeff, coefficient,
                                                                           fplot=False, penalty=penalty)

            if het_composite_obj_val > het_best_so_far:
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

            aug_X_next = my_propose_location(augmented_expected_improvement, aug_X_sample, aug_Y_sample, noise,l_init,
                                             sigma_f_init, bounds, plot_sample, n_restarts=3, min_val=300, aleatoric_weight=aleatoric_penalty, aei=True)

            # Obtain next noisy sample from the objective function
            aug_Y_next = linear_sin_noise(aug_X_next, noise_coeff, plot_sample, coefficient, fplot=False)
            aug_composite_obj_val, aug_noise_val = max_sin_noise_objective(aug_X_next, noise_coeff, coefficient, fplot=False, penalty=penalty)

            if aug_composite_obj_val > aug_best_so_far:
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

            aug_het_X_next = heteroscedastic_propose_location(heteroscedastic_augmented_expected_improvement,
                                                              aug_het_X_sample,aug_het_Y_sample, noise, l_init, sigma_f_init,
                                                              l_noise_init, sigma_f_noise_init, gp2_noise, num_iters, sample_size,
                                                              bounds, plot_sample, n_restarts=3, min_val=300, aleatoric_weight=aleatoric_penalty)

            # Obtain next noisy sample from the objective function
            aug_het_Y_next = linear_sin_noise(aug_het_X_next, noise_coeff, plot_sample, coefficient, fplot=False)
            aug_het_composite_obj_val, aug_het_noise_val = max_sin_noise_objective(aug_het_X_next, noise_coeff, coefficient, fplot=False, penalty=penalty)

            if aug_het_composite_obj_val > aug_het_best_so_far:
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

        # adding the best values in order of iteration on top of each other element wise
        rand_running_sum += np.array(rand_obj_val_list)  # just the way to average out across all random trials
        rand_squares += np.array(rand_obj_val_list) ** 2  # likewise for errors
        homo_running_sum += np.array(homo_obj_val_list)
        homo_squares += np.array(homo_obj_val_list) ** 2
        hetero_running_sum += np.array(het_obj_val_list)
        hetero_squares += np.array(het_obj_val_list) ** 2
        aug_running_sum += np.array(aug_obj_val_list)
        aug_squares += np.array(aug_obj_val_list) ** 2
        aug_het_running_sum += np.array(aug_het_obj_val_list)
        aug_het_squares += np.array(aug_het_obj_val_list) ** 2

        rand_noise_running_sum += np.array(rand_noise_val_list)  # just the way to average out across all random trials
        rand_noise_squares += np.array(rand_noise_val_list) ** 2  # likewise for errors
        homo_noise_running_sum += np.array(homo_noise_val_list)
        homo_noise_squares += np.array(homo_noise_val_list) ** 2
        hetero_noise_running_sum += np.array(het_noise_val_list)
        hetero_noise_squares += np.array(het_noise_val_list) ** 2
        aug_noise_running_sum += np.array(aug_noise_val_list)
        aug_noise_squares += np.array(aug_noise_val_list) ** 2
        aug_het_noise_running_sum += np.array(aug_het_noise_val_list)
        aug_het_noise_squares += np.array(aug_het_noise_val_list) ** 2

    rand_means = rand_running_sum / random_trials
    homo_means = homo_running_sum / random_trials
    hetero_means = hetero_running_sum / random_trials
    rand_errs = (np.sqrt(rand_squares / random_trials - rand_means ** 2))/np.sqrt(random_trials)
    homo_errs = (np.sqrt(homo_squares / random_trials - homo_means ** 2))/np.sqrt(random_trials)
    hetero_errs = (np.sqrt(hetero_squares / random_trials - hetero_means ** 2))/np.sqrt(random_trials)
    aug_means = aug_running_sum / random_trials
    aug_errs = (np.sqrt(aug_squares / random_trials - aug_means ** 2))/np.sqrt(random_trials)
    aug_het_means = aug_het_running_sum / random_trials
    aug_het_errs = (np.sqrt(aug_het_squares / random_trials - aug_het_means ** 2))/np.sqrt(random_trials)

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

    print('List of average random sampling values is: ' + str(rand_means))
    print('List of random sampling errors is:' + str(rand_errs))
    print('List of average homoscedastic values is: ' + str(homo_means))
    print('List of homoscedastic errors is: ' + str(homo_errs))
    print('List of average heteroscedastic values is ' + str(hetero_means))
    print('List of heteroscedastic errors is: ' + str(hetero_errs))
    print('List of average AEI values is: ' + str(aug_means))
    print('List of AEI errors is: ' + str(aug_errs))
    print('List of average het-AEI values is: ' + str(aug_het_means))
    print('List of het-AEI errors is: ' + str(aug_het_errs))

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

    if fill:
        plt.plot(iter_x, rand_means, color='tab:orange', label='Random Sampling')
        plt.plot(iter_x, homo_means, color='tab:blue', label='Homoscedastic')
        plt.plot(iter_x, hetero_means, color='tab:green', label='Heteroscedastic ANPEI')
        plt.plot(iter_x, aug_means, color='tab:red', label='Homoscedastic AEI')
        plt.plot(iter_x, aug_het_means, color='tab:purple', label='Heteroscedastic AEI')
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
    if penalty > 1:
        plt.ylabel(f'f(x) - {penalty}*g(x)', fontsize=14)
    else:
        plt.ylabel('f(x) - g(x)', fontsize=14)
    plt.yticks([3, 4, 5, 6, 7])
    plt.tick_params(labelsize=14)
    plt.legend(loc=4, fontsize=14)
    plt.tight_layout()
    plt.savefig('toy_figures/bayesopt_plot{}_iters_{}_random_trials_and_{}_coefficient_times_100_and_noise_coeff_times_'
               '100_of_{}_init_num_samples_of_{}_and_seed_{}_new_penalty_is_{}_aleatoric_weight_is_{}_new_aei2'.format(bayes_opt_iters, random_trials,
                                                                         int(coefficient * 100), int(noise_coeff * 100),
                                                                         init_num_samples, numpy_seed, penalty, aleatoric_penalty))

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

    if fill:
        plt.plot(iter_x, rand_noise_means, color='tab:orange', label='Random Sampling')
        plt.plot(iter_x, homo_noise_means, color='tab:blue', label='Homoscedastic')
        plt.plot(iter_x, hetero_noise_means, color='tab:green', label='Heteroscedastic ANPEI')
        plt.plot(iter_x, aug_noise_means, color='tab:red', label='Homoscedastic AEI')
        plt.plot(iter_x, aug_het_noise_means, color='tab:purple', label='Heteroscedastic AEI')
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
    plt.yticks([0.5, 1.5, 2.5, 3.5])
    plt.legend(loc=1, fontsize=14)
    plt.tight_layout()
    plt.savefig('toy_figures/bayesopt_plot{}_iters_{}_random_trials_and_{}_coefficient_times_100_and_noise_coeff_times_'
               '100_of_{}_init_num_samples_of_{}_and_seed_{}_noise_only_penalty_is_{}_aleatoric_weight_is_{}_new_aei2'.format(bayes_opt_iters, random_trials,
                                                                         int(coefficient * 100), int(noise_coeff * 100),
                                                                         init_num_samples, numpy_seed, penalty, aleatoric_penalty))