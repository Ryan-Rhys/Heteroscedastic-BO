# Copyright Lee Group 2020
# Author: Alex Aldrick
"""
This module contains the code for tuning the 1D toy_sin_noise problem and seeing the effect of changing the form of the sin function and its coefficients.
"""

import sys
from pathlib import Path # if you haven't already done so
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from acquisition_funcs.acquisition_functions import heteroscedastic_expected_improvement, heteroscedastic_propose_location, my_propose_location, my_expected_improvement, augmented_expected_improvement, heteroscedastic_augmented_expected_improvement
from objective_funcs.objective_functions import max_sin_noise_objective
import numpy as np

def linear_sin_noise(X, noise, plot_sample, coefficient, modification=False, fplot=True):
    """
    1D noise function defined where noise increases linearly in the input domain. Bounds for a bimodal function could be
    [0, 3*pi]

    :param X: input dimension
    :param noise: noise level coefficient for linearly increasing noise
    :param plot_sample: Sample for plotting purposes (points in the input domain)
    :param coefficient: Has the effect of making the maximum with larger noise larger
    :param modification: Whether to modify the function to have one maxima lower than the other
    :param fplot: Boolean indicating whether to plot the objective, samples and noise function
    :return: f(X) + noise(X)
    """

    if modification:
        linear_sin_noise = np.sin(X) + coefficient*X + (noise * np.random.randn(*X.shape) * X)
        plot_sin_function = np.sin(plot_sample) + coefficient*plot_sample
    else:
        linear_sin_noise = np.sin(X) + (noise * np.random.randn(*X.shape) * X)
        plot_sin_function = np.sin(plot_sample)

    if fplot:
        #plt.plot(X, linear_sin_noise, '+', color='green', markersize='12', linewidth='8', label='samples with Gaussian noise')
        #plt.plot(plot_sample, plot_sin_function, color='blue', label='mean of generative process')
        plt.plot(plot_sample, plot_sin_function - noise*plot_sample, color='purple')#, label='noise function')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Black-Box Objective')
        #plt.legend()
        plt.ylim(-2, 2)
        plt.xlim(0, 10)
        plt.savefig('toy_figures/black_box_tuned_multiple.png')

    return linear_sin_noise

def hetero_BO(noise_coeff):
    """
    param noise_coeff: the noise coefficient used to determine the magnitude of the noise; for example
                       if modification=True, sin(x) + 0.2x - noise_coeff*x.
    return: hetero_means: the mean values of objective values across random trials
            hetero_errs: the error associated with the objective values
            lower_hetero: magnitude of lower error boundary
            upper_hetero: magnitude of upper error boundary
            bayes_opt_iters: number of iterations of Bayesian Optimisation
            
    """
    modification = True  # Switches between sin(x) - False and sin(x) + 0.05x - True
    coefficient = 0.2  # tunes the relative size of the maxima in the function (used when modification = True)

    # Number of iterations
    random_trials = 10
    bayes_opt_iters = 5

    # We perform random trials of Bayesian Optimisation

    hetero_running_sum = np.zeros(bayes_opt_iters)
    hetero_squares = np.zeros(bayes_opt_iters)

    # We compute the objective corresponding to aleatoric noise only

    hetero_noise_running_sum = np.zeros(bayes_opt_iters)
    hetero_noise_squares = np.zeros(bayes_opt_iters)

    for i in range(random_trials):  # random trials to get average across x number of trials
        # loading bar
        print("Progress: {:2.1%}".format(i / 10), end="\r")

        #numpy_seed = i + 62
        #np.random.seed(numpy_seed)

        noise_coeff = 0.25  # noise coefficient will be noise(X) will be linear e.g. 0.2 * X
        bounds = np.array([0, 10]).reshape(-1, 1)  # bounds of the Bayesian Optimisation problem.

        #  Initial noisy data points sampled uniformly at random from the input space.

        init_num_samples = 3  # all un-named plots were 33 initial samples
        X_init = np.random.uniform(0, 10, init_num_samples).reshape(-1,
                                                                    1)  # sample 7 points at random from the bounds to initialise with
        plot_sample = np.linspace(0, 10, 50).reshape(-1, 1)  # samples for plotting purposes

        Y_init = linear_sin_noise(X_init, noise_coeff, plot_sample, coefficient, modification, fplot=False)

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

        het_best_so_far = -300
        het_obj_val_list = []
        het_noise_val_list = []

        for i in range(bayes_opt_iters):
            # number of BO iterations i.e. number of times sampled from black-box function using the acquisition function.

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

        # adding the best values in order of iteration on top of each other element wise
        # just the way to average out across all random trials
        # likewise for errors
        hetero_running_sum += np.array(het_obj_val_list)
        hetero_squares += np.array(het_obj_val_list) ** 2

    hetero_means = hetero_running_sum / random_trials
    hetero_errs = np.sqrt(hetero_squares / random_trials - hetero_means ** 2)
    lower_hetero = np.array(hetero_means) - np.array(hetero_errs)
    upper_hetero = np.array(hetero_means) + np.array(hetero_errs)

    print('List of average heteroscedastic values is ' + str(hetero_means))
    print('List of heteroscedastic errors is: ' + str(hetero_errs))

    return hetero_means, hetero_errs, lower_hetero, upper_hetero, bayes_opt_iters

if __name__ == "__main__":

    # adding sin plot function to see all current forms of function.
    for i in range(3):
        noise_coeff = 0.25 + i*0.05
        print ('checking we are in inner loop')
        modification=True
        coefficient=0.2
        # noise_coeff = 0.25  # noise coefficient will be noise(X) will be linear e.g. 0.2 *
        init_num_samples = 3  # all un-named plots were 33 initial samples
        X_init = np.random.uniform(0, 10, init_num_samples).reshape(-1, 1)  # sample 7 points at random from the bounds to initialise with
        plot_sample = np.linspace(0, 10, 50).reshape(-1, 1)  # samples for plotting purposes
        plot_sin_function = np.sin(plot_sample) + coefficient*plot_sample

        random_colour = (np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
        label = 'sin(x) - ' + str(np.round((i+1)*0.05,2)) + 'x'
        plt.plot(plot_sample, plot_sin_function - noise_coeff*plot_sample, color=random_colour, label=label)# label='noise function')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.legend(loc=4)
        plt.title('Forms of 1D function being tested')
        plt.ylim(-2, 2)
        plt.xlim(0, 10)

    plt.savefig('toy_figures/black_box_tuned_multiple.png')

    plt.figure()

    for i in range(3):
        print('Setting sin function as: sin(x) - ' + str(np.round((i+1)*0.05,2)) + 'x')
        noise_coeff = 0.25 + i*0.05
        hetero_means, hetero_errs, lower_hetero, upper_hetero, bayes_opt_iters = hetero_BO(noise_coeff)

        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        iter_x = np.arange(1, bayes_opt_iters + 1)  # need to not hard code TODO
        random_colour = (np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
        label = 'sin(x) - ' + str(np.round((i+1)*0.05,2)) + 'x'
        plt.plot(iter_x, hetero_means, color=random_colour, label=label)
        lower_hetero = np.array(hetero_means) - np.array(hetero_errs)
        upper_hetero = np.array(hetero_means) + np.array(hetero_errs)
        plt.fill_between(iter_x, lower_hetero, upper_hetero, color=random_colour, alpha=0.1)

    ax.title.set_fontsize(10)
    plt.title('Tuning the 1D sin function')
    plt.xlabel('Number of Function Evaluations')
    plt.ylabel('Objective Function Value - Noise')
    plt.legend(loc=4)
    plt.savefig('toy_figures/tune_1D_looped.png')



