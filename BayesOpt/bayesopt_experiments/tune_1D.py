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
import numpy as np


def objective(X, noise_coeff):
    """
    1D noise function defined where noise increases linearly in the input domain.
    :param X: input dimension
    :param noise_coeff: noise level coefficient for linearly increasing noise
    :return: f(X) + sampling_noise(X)
    """

    objective = -1*np.power(X, 5) + -1.3*np.power(X, 4) + 6.2*np.power(X, 3) + 5.8*np.power(X, 2) + -3*X - (noise_coeff * np.random.randn(*X.shape) * X) + 0.3

    return objective


def max_objective(X, noise_coeff):
    """
    Objective function for maximising objective - aleatoric noise for the sin wave with linear noise. Used for
    monitoring the best value in the optimisation obtained so far.

    :param X: input to evaluate objective; can be an array of values
    :param noise: noise level coefficient
    :return: value of the black-box objective that penalises aleatoric noise, value of the noise at X
    """

    noise_value = noise_coeff * X  # value of the heteroscedastic noise at the point(s) X
    objective_value = -1*np.power(X, 5) + -1.3*np.power(X, 4) + 6.2*np.power(X, 3) + 5.8*np.power(X, 2) + -3*X + 0.3
    composite_objective = objective_value - noise_value

    composite_objective = float(composite_objective)
    noise_value = float(noise_value)

    return composite_objective, noise_value


def homo_BO(noise_coeff, x_lower_bound, x_upper_bound):
    """
    param noise_coeff: the noise coefficient used to determine the magnitude of the noise; for example, if modification=True, sin(x) + 0.2x - noise_coeff*x
    param coefficient: coefficient of x i.e. 0.2 in the line above
    param x_lower_bound: lower bound of x to consider for BO and for plotting
    param x_upper_bound: upper bound of x to consider for BO and for plotting
    return: homo_means: the mean values of objective values across random trials
            homo_errs: the error associated with the objective values
            lower_homo: magnitude of lower error boundary
            upper_homo: magnitude of upper error boundary
            homo_X_sample: list of X values sampled
            homo_Y_sample: list of corresponding Y values at points in homo_X_sample
            bayes_opt_iters: number of bayes optimisation iterations
    """

    # Number of iterations
    random_trials = 10
    bayes_opt_iters = 10

    # We perform random trials of Bayesian Optimisation

    homo_running_sum = np.zeros(bayes_opt_iters)
    homo_squares = np.zeros(bayes_opt_iters)


    for i in range(random_trials):  # random trials to get average across x number of trials
        # loading bar
        print("Progress: {:2.1%}".format(i / 10), end="\r")

        bounds = np.array([x_lower_bound, x_upper_bound]).reshape(-1, 1)  # bounds of the Bayesian Optimisation problem.

        #  Initial noisy data points sampled uniformly at random from the input space.

        init_num_samples = 3  # all un-named plots were 3 initial samples
        X_init = np.random.uniform(x_lower_bound, x_upper_bound, init_num_samples).reshape(-1,1)  # sample 7 points at random from the bounds to initialise with
        plot_sample = np.linspace(x_lower_bound, x_upper_bound, 50).reshape(-1, 1)  # samples for plotting purposes

        Y_init = objective(X_init, noise_coeff)

        homo_X_sample = X_init.reshape(-1, 1)
        homo_Y_sample = Y_init.reshape(-1, 1)
        # initial GP hypers

        l_init = 1.0
        sigma_f_init = 1.0
        noise = 1.0  # need to be careful about how we set this because it's not currently being optimised in the code (see reviewer comment)
        l_noise_init = 1.0
        sigma_f_noise_init = 1.0
        gp2_noise = 1.0
        num_iters = 10
        sample_size = 100

        homo_best_so_far = -300
        homo_obj_val_list = []

        for i in range(bayes_opt_iters):

            current_iter = i

            homo_X_next = my_propose_location(my_expected_improvement, homo_X_sample, homo_Y_sample, noise, l_init, sigma_f_init, bounds, plot_sample, current_iter, n_restarts=3, min_val=300)
            homo_Y_next = objective(homo_X_next, noise_coeff)
            homo_composite_obj_val, homo_noise_val = max_objective(homo_X_next, noise_coeff)
            # if the new Y-value is better than our best so far, save it as best so far and append it to best Y-values list in *_composite_obj_val
            if homo_composite_obj_val > homo_best_so_far:
                homo_best_so_far = homo_composite_obj_val
                homo_obj_val_list.append(homo_composite_obj_val)
            else:
                homo_obj_val_list.append(homo_best_so_far)

            # Add sample to previous samples
            homo_X_sample = np.vstack((homo_X_sample, homo_X_next))
            homo_Y_sample = np.vstack((homo_Y_sample, homo_Y_next))

        # adding the best values in order of iteration on top of each other element wise
        # just the way to average out across all random trials
        # likewise for errors
        homo_running_sum += np.array(homo_obj_val_list)
        homo_squares += np.array(homo_obj_val_list) ** 2

    homo_means = homo_running_sum / random_trials
    homo_errs = np.sqrt(homo_squares / random_trials - homo_means ** 2)
    lower_homo = np.array(homo_means) - np.array(homo_errs)
    upper_homo = np.array(homo_means) + np.array(homo_errs)

    print('List of average homoscedastic values is ' + str(homo_means))
    print('List of homoscedastic errors is: ' + str(homo_errs))

    return homo_means, homo_errs, lower_homo, upper_homo, homo_X_sample, homo_Y_sample, bayes_opt_iters


def hetero_BO(noise_coeff, x_lower_bound, x_upper_bound):
    """
    param noise_coeff: the noise coefficient used to determine the magnitude of the noise; for example, if modification=True, sin(x) + 0.2x - noise_coeff*x
    param x_lower_bound: lower bound of x to consider for BO and for plotting
    param x_upper_bound: upper bound of x to consider for BO and for plotting
    return: hetero_means: the mean values of objective values across random trials
            hetero_errs: the error associated with the objective values
            lower_hetero: magnitude of lower error boundary
            upper_hetero: magnitude of upper error boundary
            het_X_sample: list of X values sampled
            het_Y_sample: list of corresponding Y values at points in het_X_sample
            bayes_opt_iters: number of bayes optimisation iterations
    """
    # Number of iterations
    random_trials = 10
    bayes_opt_iters = 10

    # We perform random trials of Bayesian Optimisation

    hetero_running_sum = np.zeros(bayes_opt_iters)
    hetero_squares = np.zeros(bayes_opt_iters)

    # We compute the objective corresponding to aleatoric noise only

    for i in range(random_trials):  # random trials to get average across x number of trials
        # loading bar
        print("Progress: {:2.1%}".format(i / 10), end="\r")

        bounds = np.array([x_lower_bound, x_upper_bound]).reshape(-1, 1)  # bounds of the Bayesian Optimisation problem.

        #  Initial noisy data points sampled uniformly at random from the input space.

        init_num_samples = 3  # all un-named plots were 3 nitial samples
        X_init = np.random.uniform(x_lower_bound, x_upper_bound, init_num_samples).reshape(-1, 1)  # sample 7 points at random from the bounds to initialise with
        plot_sample = np.linspace(x_lower_bound, x_upper_bound, 50).reshape(-1, 1)  # samples for plotting purposes

        Y_init = objective(X_init, noise_coeff)

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

        for i in range(bayes_opt_iters):

            current_iter = i
            # number of BO iterations i.e. number of times sampled from black-box function using the acquisition function.

            # Obtain next sampling point from the het acquisition function (ANPEI)

            het_X_next = heteroscedastic_propose_location(heteroscedastic_expected_improvement, het_X_sample,
                                                          het_Y_sample, noise, l_init, sigma_f_init, l_noise_init,
                                                          sigma_f_noise_init, gp2_noise, num_iters, sample_size, bounds,
                                                          plot_sample, current_iter, n_restarts=3, min_val=300)

            # Obtain next noisy sample from the objective function
            het_Y_next = objective(het_X_next, noise_coeff)
            het_composite_obj_val, het_noise_val = max_objective(het_X_next, noise_coeff)

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

    return hetero_means, hetero_errs, lower_hetero, upper_hetero, het_X_sample, het_Y_sample, bayes_opt_iters


def random_sampling(noise_coeff, x_lower_bound, x_upper_bound):
    """
    param noise_coeff: the noise coefficient used to determine the magnitude of the noise; for example, if modification=True, sin(x) + 0.2x - noise_coeff*x
    param coefficient: coefficient of x i.e. 0.2 in the line above
    param x_lower_bound: lower bound of x to consider for BO and for plotting
    param x_upper_bound: upper bound of x to consider for BO and for plotting
    return: rand_means: the mean values of objective values across random trials
            rand_errs: the error associated with the objective values
            lower_rand: magnitude of lower error boundary
            upper_rand: magnitude of upper error boundary
    """
    # We perform random trials of Bayesian Optimisation

    random_trials = 10
    rand_iters = 10

    rand_running_sum = np.zeros(rand_iters)
    rand_squares = np.zeros(rand_iters)

    for i in range(random_trials):
        # loading bar
        print("Progress: {:2.1%}".format(i / 10), end="\r")

        bounds = np.array([x_lower_bound, x_upper_bound]).reshape(-1, 1)  # bounds of the Bayesian Optimisation problem.

        #  Initial noisy data points sampled uniformly at random from the input space.

        init_num_samples = 3  # all un-named plots were 3 nitial samples
        X_init = np.random.uniform(x_lower_bound, x_upper_bound, init_num_samples).reshape(-1,
                                                                    1)  # sample 7 points at random from the bounds to initialise with
        plot_sample = np.linspace(x_lower_bound, x_upper_bound, 50).reshape(-1, 1)  # samples for plotting purposes

        Y_init = objective(X_init, noise_coeff)

        rand_X_sample = X_init.reshape(-1, 1)
        rand_Y_sample = Y_init.reshape(-1, 1)
        # initial GP hypers

        l_init = 1.0
        sigma_f_init = 1.0
        noise = 1.0  # need to be careful about how we set this because it's not currently being optimised in the code (see reviewer comment)
        l_noise_init = 1.0
        sigma_f_noise_init = 1.0
        gp2_noise = 1.0
        num_iters = 10
        sample_size = 100

        rand_best_so_far = -300
        rand_obj_val_list = []
        rand_noise_val_list = []

        for i in range(rand_iters):
            # number of BO iterations i.e. number of times sampled from black-box function using the acquisition function.
            rand_X_next = np.random.uniform(x_lower_bound, x_upper_bound)  # this just takes X not the sin function itself
            # check if random point's Y value is better than best so far
            rand_composite_obj_val, rand_noise_val = max_objective(rand_X_next, noise_coeff)
            if rand_composite_obj_val > rand_best_so_far:
                rand_best_so_far = rand_composite_obj_val
                rand_obj_val_list.append(rand_composite_obj_val)
            else:
                rand_obj_val_list.append(rand_best_so_far)
            # if yes, save it, if no, save best so far into list of best y-value per iteration in rand_composite_obj_val

        rand_running_sum += np.array(rand_obj_val_list)  # just the way to average out across all random trials
        rand_squares += np.array(rand_obj_val_list) ** 2  # likewise for errors

    rand_means = rand_running_sum / random_trials
    rand_errs = np.sqrt(rand_squares / random_trials - rand_means ** 2)

    return rand_means, rand_errs, x_lower_bound, x_upper_bound, rand_iters

if __name__ == "__main__":

    no_of_tests = 1  # number of noise_coeffs to cycle through and plot
    # adding sin plot function to see all current forms of function.
    x_lower_bound = -2
    x_upper_bound = 2
    noise_coeff = 10.6  #have to make negative here to subtract the noise rate function!1

    for i in range(no_of_tests):  # plotter for fiddled 1D functions
        noise_coeff += i*0.05  # change this to fiddle with 1D function
        plot_sample = np.linspace(x_lower_bound, x_upper_bound, 50).reshape(-1, 1)  # samples for plotting purposes
        plot_sin_function = -1*np.power(plot_sample, 5) + -1.3*np.power(plot_sample, 4) + 6.2*np.power(plot_sample, 3) + 5.8*np.power(plot_sample, 2) + -3*plot_sample + 0.3
        plot_sin_function_noise = plot_sin_function - noise_coeff*plot_sample

        #random_colour = (np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
        label='-1*X^5 + -1.3*X^4 + 6.2*X^3 + 5.8*X^2 + -3*X + 0.3'
        plt.plot(plot_sample, plot_sin_function, color='b', label=label)
        label='-1*X^5 + -1.3*X^4 + 6.2*X^3 + 5.8*X^2 + -13.6*X + 0.3'
        plt.plot(plot_sample, plot_sin_function_noise, color='r', label=label)
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.legend(loc=4)
        plt.title('Forms of 1D function being tested')
        plt.autoscale(enable=True, axis='y')
        #plt.ylim(0, 11)
        plt.xlim(x_lower_bound, x_upper_bound)

    print('Saving 1D functional forms in toy_figures/black_box_tuned_multiple.png')
    #plt.autoscale(enable=True)
    #plt.axhline(y=0, color='black', linewidth=.5)
    #plt.xlim(left=0)
    plt.savefig('toy_figures/black_box_tuned_multiple.png')

    plt.figure()

    for i in range(no_of_tests):
        print('i is:' + str(i))
        noise_coeff += i*0.05  # change this to fiddle with 1D function

        hetero_means, hetero_errs, lower_hetero, upper_hetero, het_X_sample, het_Y_sample, bayes_opt_iters = hetero_BO(noise_coeff, x_lower_bound, x_upper_bound)

        homo_means, homo_errs, lower_homo, upper_homo, homo_X_sample, homo_Y_sample, bayes_opt_iters = homo_BO(noise_coeff, x_lower_bound, x_upper_bound)

        #rand_means, rand_errs, lower_rand, upper_rand, rand_iters = random_sampling(noise_coeff, x_lower_bound, x_upper_bound)

        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        iter_x = np.arange(1, bayes_opt_iters + 1)
        random_colour = (np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
        label='Homoscedastic'
        plt.plot(iter_x, homo_means, color=random_colour, label=label)
        lower_homo = np.array(homo_means) - np.array(homo_errs)
        upper_hetero = np.array(homo_means) + np.array(homo_errs)
        plt.fill_between(iter_x, lower_homo, upper_homo, color=random_colour, alpha=0.1)
        random_colour = (np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
        label='Heteroscedastic ANPEI'
        plt.plot(iter_x, hetero_means, color=random_colour, label=label)
        lower_hetero = np.array(hetero_means) - np.array(hetero_errs)
        upper_hetero = np.array(hetero_means) + np.array(hetero_errs)
        plt.fill_between(iter_x, lower_hetero, upper_hetero, color=random_colour, alpha=0.1)
#        iter_rand_x = np.arange(1, rand_iters + 1)
#        random_colour = (np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
#        plt.plot(iter_rand_x, rand_means, color=random_colour, label='Random Sampling')
#        lower_rand = np.array(rand_means) - np.array(rand_errs)
#        upper_rand = np.array(rand_means) + np.array(rand_errs)
#        plt.fill_between(iter_rand_x, lower_rand, upper_rand, color=random_colour, alpha=0.1)

    ax.title.set_fontsize(10)
    plt.title('Tuning the 1D sin function')
    plt.xlabel('Number of Function Evaluations')
    plt.ylabel('Objective Function Value - Noise')
    plt.legend(loc=4)
    print('Saving performance test in toy_figures/tune_1D_looped.png')
    plt.savefig('toy_figures/tune_1D_looped.png')
