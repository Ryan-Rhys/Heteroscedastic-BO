# Author: Ryan-Rhys Griffiths
"""
This module contains the code for benchmarking heteroscedastic Bayesian Optimisation on the 2D Branin-Hoo Function
with heteroscedastic noise maximisation.
"""

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

from acquisition_funcs.acquisition_functions import heteroscedastic_one_off_expected_improvement, heteroscedastic_propose_location, \
    my_propose_location, my_expected_improvement, augmented_expected_improvement, heteroscedastic_one_off_augmented_expected_improvement
from objective_funcs.objective_functions import one_off_min_branin_noise_function, heteroscedastic_branin


if __name__ == '__main__':

    standardised = True
    penalty = 5
    fill = True  # Whether to plot errorbars as fill or not.
    plot_collected = True # Whether to plot collected data points

    # Number of iterations
    bayes_opt_iters = 10
    random_trials = 10

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

        numpy_seed = i + 50
        np.random.seed(numpy_seed)

        if standardised:
            bounds = np.array([[0, 1.0], [0, 1.0]])
        else:
            bounds = np.array([[-5.0, 10.0], [0.0, 15.0]])  # bounds of the Bayesian Optimisation problem.

        #  Initial noisy data points sampled uniformly at random from the input space.

        grid_size = 4

        if standardised:
            x1 = np.random.uniform(0, 1, size=(grid_size,))
            x2 = np.random.uniform(0, 1, size=(grid_size,))
        else:
            x1 = np.random.uniform(-5.0, 10.0, size=(grid_size,))
            x2 = np.random.uniform(0.0, 15.0, size=(grid_size,))

        X_init = np.array(np.meshgrid(x1, x2)).T.reshape(-1, 2)
        Y_init = heteroscedastic_branin(X_init[:, 0], X_init[:, 1])

        if standardised:
            x1_star = np.arange(0, 1, 0.05)
            x2_star = np.arange(0, 1, 0.05)
        else:
            x1_star = np.arange(-5.0, 10.0, 0.5)
            x2_star = np.arange(0.0, 15.0, 0.5)

        plot_sample = np.array(np.meshgrid(x1_star, x2_star)).T.reshape(-1, 2)  # Where 2 gives the dimensionality

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

        rand_best_so_far = 300
        homo_best_so_far = 300  # value to beat
        het_best_so_far = 300
        aug_best_so_far = 300
        aug_het_best_so_far = 300
        rand_noise_best_so_far = -300  # value to beat
        homo_noise_best_so_far = -300
        het_noise_best_so_far = -300
        aug_noise_best_so_far = -300
        aug_het_noise_best_so_far = -300
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

        for i in range(bayes_opt_iters):

            print(i)

            # random sampling baseline

            if standardised:
                random_x1_next = np.random.uniform(0, 1, size=(1,))
                random_x2_next = np.random.uniform(0, 1, size=(1,))
            else:
                random_x1_next = np.random.uniform(-5.0, 10.0, size=(1,))
                random_x2_next = np.random.uniform(0.0, 15.0, size=(1,))
            random_X_next = np.array(np.meshgrid(random_x1_next, random_x2_next)).T.reshape(-1, 2)

            rand_collected_x1.append(random_x1_next)
            rand_collected_x2.append(random_x2_next)

            f_plot = True
            if i > 0:
                f_plot=False

            random_Y_next = heteroscedastic_branin(random_x1_next, random_x2_next, standardised=standardised, f_plot=f_plot)
            random_composite_obj_val, rand_noise_val = one_off_min_branin_noise_function(random_x1_next, random_x2_next, standardised, penalty)

            f_plot = False

            if random_composite_obj_val < rand_best_so_far:
                rand_best_so_far = random_composite_obj_val
                rand_obj_val_list.append(random_composite_obj_val)
            else:
                rand_obj_val_list.append(rand_best_so_far)

            if rand_noise_val > rand_noise_best_so_far:
                rand_noise_best_so_far = rand_noise_val
                rand_noise_val_list.append(rand_noise_val)
            else:
                rand_noise_val_list.append(rand_noise_best_so_far)

            # Obtain next sampling point from the acquisition function (expected_improvement)

            homo_X_next = my_propose_location(my_expected_improvement, homo_X_sample, homo_Y_sample, noise, l_init, sigma_f_init,
                                              bounds, plot_sample, n_restarts=5, min_val=6000)

            homo_collected_x1.append(homo_X_next[:, 0])
            homo_collected_x2.append(homo_X_next[:, 1])

            # Obtain next noisy sample from the objective function
            homo_Y_next = heteroscedastic_branin(homo_X_next[:, 0], homo_X_next[:, 1], standardised=standardised, f_plot=f_plot, penalty=penalty)
            homo_composite_obj_val, homo_noise_val = one_off_min_branin_noise_function(homo_X_next[:, 0], homo_X_next[:, 1], standardised, penalty)

            if homo_composite_obj_val < homo_best_so_far:
                homo_best_so_far = homo_composite_obj_val
                homo_obj_val_list.append(homo_composite_obj_val)
            else:
                homo_obj_val_list.append(homo_best_so_far)

            if homo_noise_val > homo_noise_best_so_far:
                homo_noise_best_so_far = homo_noise_val
                homo_noise_val_list.append(homo_noise_val)
            else:
                homo_noise_val_list.append(homo_noise_best_so_far)

            # Add sample to previous samples
            homo_X_sample = np.vstack((homo_X_sample, homo_X_next))
            homo_Y_sample = np.vstack((homo_Y_sample, homo_Y_next))

            # Obtain next sampling point from the het acquisition function (ANPEI)

            het_X_next = heteroscedastic_propose_location(heteroscedastic_one_off_expected_improvement, het_X_sample,
                                                          het_Y_sample, noise, l_init, sigma_f_init, l_noise_init,
                                                          sigma_f_noise_init, gp2_noise, num_iters, sample_size, bounds,
                                                          plot_sample, n_restarts=5, min_val=6000)

            het_collected_x1.append(het_X_next[:, 0])
            het_collected_x2.append(het_X_next[:, 1])

            # Obtain next noisy sample from the objective function
            het_Y_next = heteroscedastic_branin(het_X_next[:, 0], het_X_next[:, 1], standardised=standardised, f_plot=f_plot, penalty=penalty)
            het_composite_obj_val, het_noise_val = one_off_min_branin_noise_function(het_X_next[:, 0], het_X_next[:, 1], standardised, penalty)

            if het_composite_obj_val < het_best_so_far:
                het_best_so_far = het_composite_obj_val
                het_obj_val_list.append(het_composite_obj_val)
            else:
                het_obj_val_list.append(het_best_so_far)

            if het_noise_val > het_noise_best_so_far:
                het_noise_best_so_far = het_noise_val
                het_noise_val_list.append(het_noise_val)
            else:
                het_noise_val_list.append(het_noise_best_so_far)

            # Add sample to previous samples
            het_X_sample = np.vstack((het_X_sample, het_X_next))
            het_Y_sample = np.vstack((het_Y_sample, het_Y_next))

            # Obtain next sampling point from the augmented expected improvement (AEI)

            aug_X_next = my_propose_location(augmented_expected_improvement, aug_X_sample, aug_Y_sample, noise, l_init, sigma_f_init,
                                             bounds, plot_sample, n_restarts=5, min_val=6000)

            aug_collected_x1.append(aug_X_next[:, 0])
            aug_collected_x2.append(aug_X_next[:, 1])

            # Obtain next noisy sample from the objective function
            aug_Y_next = heteroscedastic_branin(aug_X_next[:, 0], aug_X_next[:, 1], standardised=standardised, f_plot=f_plot, penalty=penalty)
            aug_composite_obj_val, aug_noise_val = one_off_min_branin_noise_function(aug_X_next[:, 0], aug_X_next[:, 1], standardised, penalty)

            if aug_composite_obj_val < aug_best_so_far:
                aug_best_so_far = aug_composite_obj_val
                aug_obj_val_list.append(aug_composite_obj_val)
            else:
                aug_obj_val_list.append(aug_best_so_far)

            if aug_noise_val > aug_noise_best_so_far:
                aug_noise_best_so_far = aug_noise_val
                aug_noise_val_list.append(aug_noise_val)
            else:
                aug_noise_val_list.append(aug_noise_best_so_far)

            # Add sample to previous sample
            aug_X_sample = np.vstack((aug_X_sample, aug_X_next))
            aug_Y_sample = np.vstack((aug_Y_sample, aug_Y_next))

            # Obtain next sampling point from the heteroscedastic augmented expected improvement (het-AEI)

            aug_het_X_next = heteroscedastic_propose_location(heteroscedastic_one_off_augmented_expected_improvement, aug_het_X_sample,
                                                          aug_het_Y_sample, noise, l_init, sigma_f_init, l_noise_init,
                                                          sigma_f_noise_init, gp2_noise, num_iters, sample_size, bounds,
                                                          plot_sample, n_restarts=5, min_val=6000)

            aug_het_collected_x1.append(aug_het_X_next[:, 0])
            aug_het_collected_x2.append(aug_het_X_next[:, 1])

            # Obtain next noisy sample from the objective function
            aug_het_Y_next = heteroscedastic_branin(aug_het_X_next[:, 0], aug_het_X_next[:, 1], standardised=standardised, f_plot=f_plot, penalty=penalty)
            aug_het_composite_obj_val, aug_het_noise_val = one_off_min_branin_noise_function(aug_het_X_next[:, 0], aug_het_X_next[:, 1], standardised, penalty)

            if aug_het_composite_obj_val < aug_het_best_so_far:
                aug_het_best_so_far = aug_het_composite_obj_val
                aug_het_obj_val_list.append(aug_het_composite_obj_val)
            else:
                aug_het_obj_val_list.append(aug_het_best_so_far)

            if aug_het_noise_val > aug_het_noise_best_so_far:
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

    rand_means = rand_running_sum / random_trials
    rand_errs = (np.sqrt(rand_squares / random_trials - rand_means **2, dtype=np.float64))/np.sqrt(random_trials)
    homo_means = homo_running_sum / random_trials
    hetero_means = hetero_running_sum / random_trials
    homo_errs = (np.sqrt(homo_squares / random_trials - homo_means ** 2, dtype=np.float64))/np.sqrt(random_trials)
    hetero_errs = (np.sqrt(hetero_squares / random_trials - hetero_means ** 2, dtype=np.float64))/np.sqrt(random_trials)
    aug_means = aug_running_sum / random_trials
    aug_errs = (np.sqrt(aug_squares / random_trials - aug_means ** 2, dtype=np.float64))/np.sqrt(random_trials)
    aug_het_means = aug_het_running_sum / random_trials
    aug_het_errs = (np.sqrt(aug_het_squares / random_trials - aug_het_means ** 2, dtype=np.float64))/np.sqrt(random_trials)

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

    plt.title('Best Objective Function Value Found so Far')
    plt.xlabel('Number of Function Evaluations')
    plt.ylabel('Objective Function Value - Noise')
    plt.tick_params(labelsize=14)
    plt.legend(loc=1)
    plt.savefig('one_off_branin_figures/bayesopt_plot{}_iters_{}_random_trials_and_grid_size_of_{}_and_seed_{}_new_standardised_is_{}_penalty_is_{}'.
                format(bayes_opt_iters, random_trials, grid_size, numpy_seed, standardised, penalty))

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

    best_noise_plot = 3*np.ones(len(iter_x))

    if fill:
        plt.plot(iter_x, best_noise_plot, '--', color='k', label='Optimal')
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

    plt.title('Highest Aleatoric Noise Found so Far')
    plt.xlabel('Number of Function Evaluations')
    plt.ylabel('Aleatoric Noise')
    plt.tick_params(labelsize=14)
    plt.legend(loc=4)
    plt.savefig('one_off_branin_figures/bayesopt_plot{}_iters_{}_random_trials_and_grid_size_of_{}_and_seed_{}_noise_only_standardised_is_{}_penalty_is_{}'.
                format(bayes_opt_iters, random_trials, grid_size, numpy_seed, standardised, penalty))
