# Copyright Lee Group 2019
# Author: Ryan-Rhys Griffiths
"""
This script performs Heteroscedastic Bayesian Optimisation for a Neural Architecture Search Classification task.
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow import set_random_seed

from acquisition_functions import my_expected_improvement, my_propose_location, heteroscedastic_propose_location, \
    heteroscedastic_expected_improvement
from exp_utils import measure_class_performance


if __name__ == '__main__':

    bayes_opt_iters = 20  # Number of iterations of Bayesian Optimisation

    # Initialise lists to compute the empirical mean and variance across the random trials with a single pass through
    # the data

    homo_running_sum = np.zeros(bayes_opt_iters)
    homo_squares = np.zeros(bayes_opt_iters)  # Following the single-pass estimator given on pg. 192 of mathematics for machine learning
    hetero_running_sum = np.zeros(bayes_opt_iters)
    hetero_squares = np.zeros(bayes_opt_iters)

    # We compute the objective corresponding to aleatoric noise only

    homo_noise_running_sum = np.zeros(bayes_opt_iters)
    homo_noise_squares = np.zeros(bayes_opt_iters)  # Following the single-pass estimator given on pg. 192 of mathematics for machine learning
    hetero_noise_running_sum = np.zeros(bayes_opt_iters)
    hetero_noise_squares = np.zeros(bayes_opt_iters)

    # We perform random trials of Bayesian Optimisation

    random_trials = 10

    for i in range(random_trials):

        numpy_seed = i + 50
        tf_seed = i + 51
        np.random.seed(numpy_seed)
        set_random_seed(tf_seed)

        dataset = load_breast_cancer()
        data, target = dataset.data, dataset.target
        data_scaler = MinMaxScaler()
        data = data_scaler.fit_transform(data)

        target = target.reshape(-1, 1)
        n_inputs = data.shape[1]
        n_outputs = target.shape[1]

        #init_neurons = [1, 100, 200, 300, 400, 500]
        init_neurons = list(np.arange(1, 500, 1))
        num_layers = 3
        k_folds = 3
        n_trials = 3
        noise_penalty = 10  # multiplicative factor by which to penalise noisy solutions

        Y_sample = []
        Y_plot_sample = []  # mean performance for plotting purposes
        noise_sample = []
        init_point_obj_vals = []

        for num_neurons in init_neurons:

            mean_performance, noise_std = measure_class_performance(data, target, num_neurons, num_layers, k_folds, n_trials)

            # We keep track of the best objective function value in the training data.
            obj_val = mean_performance - noise_penalty*noise_std
            init_point_obj_vals.append(obj_val)

            performance = np.random.normal(mean_performance, noise_std)
            Y_sample.append(performance)
            Y_plot_sample.append(mean_performance)
            noise_sample.append(noise_std)

        best_so_far = np.max(init_point_obj_vals)  # we initialise the best value to be the best value in initialisation
        print('Best objective function value in the initialisation is: ' + str(best_so_far))

        best_noise_so_far = np.min(noise_sample)

        X_sample = np.array(init_neurons).reshape(-1, 1)
        Y_sample = np.array(Y_sample).reshape(-1, 1)
        Y_plot_sample = np.array(Y_plot_sample).reshape(-1, 1)

        # Plot the initial samples

        f_plot_samples = True

        if f_plot_samples:

            # Add error bars

            noise_sample = np.array(noise_sample).reshape(-1, 1)

            ax = plt.gca()
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            #plt.scatter(X_sample, Y_sample, marker='+')
            lower = Y_plot_sample - 2 * noise_sample
            upper = Y_plot_sample + 2 * noise_sample
            yerr = upper - lower
            #plt.errorbar(X_sample, Y_sample, yerr=yerr, fmt='+', ecolor='r')
            plt.errorbar(X_sample, Y_plot_sample, yerr=yerr, ecolor='r')
            plt.xlabel('Number of Neurons per Layer')
            plt.ylabel('f(x)')
            plt.title('Black Box Function {} Folds and {} Trials'.format(k_folds, n_trials))
            plt.savefig('figures/black_box_function_{}_folds_and_{}_trials_{}_upper_neurons_and_numlayers{}'.format(k_folds, n_trials, max(init_neurons), num_layers))
            plt.show()

        # Standardise inputs and outputs

        X_scaler = StandardScaler().fit(X_sample)
        Y_scaler = StandardScaler().fit(Y_sample)
        X_sample = X_scaler.transform(X_sample)
        Y_sample = Y_scaler.transform(Y_sample)

        # Initial GP hyperparameters

        l_init = 1
        sigma_f_init = 1
        noise = 1

        # Extra Heteroscedastic GP hyperparameters

        l_noise_init = 1
        sigma_f_noise_init = 1
        gp2_noise = 1
        num_iters = 10
        sample_size = 100

        # Bounds on the input space

        bounds = np.array([1, 500])
        plot_sample = np.linspace(bounds[0], bounds[1], bounds[1]).reshape(-1, 1) # for plotting the predictive mean and variance.

        plot_sample = X_scaler.transform(plot_sample)  # we standardise the plotting locations.

        # We set the experiment configuration

        obj_val_list_homo = []  # we store the objective function values
        obj_val_list_het = []  # we store the objective function values for heteroscedastic BO
        noise_list_homo = []  # we store the ground truth aleatoric noise for homoscedastic BO
        noise_list_het = []  # we store the ground truth aleatoric noise for heteroscedastic BO
        best_neurons_homo = None  # we keep track of the best number of neurons under the mean performance - noise objective for homoscedastic BO
        best_neurons_het = None
        collected_homoscedastic_neurons = []  # Queried locations in neuron space for homoscedastic BO
        collected_heteroscedastic_neurons = []  # Queried locations in neuron space of heteroscedatic BO
        X_sample_homo, Y_sample_homo = X_sample.copy(), Y_sample.copy()  # We initialise the X locations
        X_sample_het, Y_sample_het = X_sample.copy(), Y_sample.copy()  # We initialise the Y values
        best_so_far_homo = best_so_far
        best_so_far_het = best_so_far
        best_noise_so_far_homo = best_noise_so_far
        best_noise_so_far_hetero = best_noise_so_far

        for iteration in range(bayes_opt_iters):

            print('Iteration number is ' + str(iteration))

            # We run Homoscedastic BO

            X_next_homo = my_propose_location(my_expected_improvement, X_sample_homo, Y_sample_homo, noise, l_init, sigma_f_init,
                                         bounds, plot_sample, n_restarts=3, min_val=300)
            X_next_homo = int(np.round(X_next_homo))  # Continuous values won't be accepted for the number of neurons.
            collected_homoscedastic_neurons.append(X_next_homo)  # append the real rather than the standardised value.

            mean_performance_homo, noise_std_homo = measure_class_performance(data, target, X_next_homo, num_layers, k_folds, n_trials)
            performance_homo = np.random.normal(mean_performance_homo, noise_std_homo)

            # Obtain next noisy sample from the objective function
            Y_next_homo = performance_homo

            # Measure objective function value
            obj_val_homo = mean_performance_homo - noise_penalty*noise_std_homo
            print('objective function value is: ' + str(obj_val_homo))

            if obj_val_homo > best_so_far_homo:
                best_so_far_homo = obj_val_homo
                obj_val_list_homo.append(obj_val_homo)
                best_neurons_homo = X_next_homo
            else:
                obj_val_list_homo.append(best_so_far_homo)

            if noise_std_homo < best_noise_so_far_homo:
                best_noise_so_far_homo = noise_std_homo
                noise_list_homo.append(noise_std_homo)
            else:
                noise_list_homo.append(best_noise_so_far_homo)

            X_next_homo = X_scaler.transform(np.array(X_next_homo).reshape(-1, 1))  # standardising the x-value as well.
            Y_next_homo = Y_scaler.transform(np.array(Y_next_homo).reshape(-1, 1))  # standardise the collected output.

            # Add sample to previous samples
            X_sample_homo = np.vstack((X_sample_homo, X_next_homo))
            Y_sample_homo = np.vstack((Y_sample_homo, Y_next_homo))

            # We run Heteroscedastic BO

            X_next_het = heteroscedastic_propose_location(heteroscedastic_expected_improvement, X_sample_het, Y_sample_het,
                                                          noise, l_init, sigma_f_init, l_noise_init, sigma_f_noise_init,
                                                          gp2_noise, num_iters, sample_size, bounds, plot_sample,
                                                          n_restarts=3, min_val=300)

            X_next_het = int(np.round(X_next_het))  # Continuous values won't be accepted for the number of neurons.
            collected_heteroscedastic_neurons.append(X_next_het)

            mean_performance_het, noise_std_het = measure_class_performance(data, target, X_next_het, num_layers, k_folds, n_trials)
            performance_het = np.random.normal(mean_performance_het, noise_std_het)

            # Obtain next noisy sample from the objective function
            Y_next_het = performance_het

            # Measure objective function value
            obj_val_het = mean_performance_het - noise_penalty*noise_std_het
            print('objective function value is: ' + str(obj_val_het))

            if obj_val_het > best_so_far_het:
                best_so_far_het = obj_val_het
                obj_val_list_het.append(obj_val_het)
                best_neurons_het = X_next_het
            else:
                obj_val_list_het.append(best_so_far_het)

            if noise_std_het < best_noise_so_far_hetero:
                best_noise_so_far_hetero = noise_std_het
                noise_list_het.append(noise_std_het)
            else:
                noise_list_het.append(best_noise_so_far_hetero)

            X_next_het = X_scaler.transform(np.array(X_next_het).reshape(-1, 1))  # we standardise the next sample
            Y_next_het = Y_scaler.transform(np.array(Y_next_het).reshape(-1, 1))

            # Add sample to previous samples
            X_sample_het = np.vstack((X_sample_het, X_next_het))
            Y_sample_het = np.vstack((Y_sample_het, Y_next_het))

        print('List of Homoscedastic objective function values of collected points is: ' + str(obj_val_list_homo))
        print('Collected numbers of Homoscedastic neurons are: ' + str(X_sample_homo))
        print('Best Number of Homoscedastic Neurons is: ' + str(best_neurons_homo))
        print('List of Heteroscedastic objective function values of collected points is: ' + str(obj_val_list_het))
        print('Collected numbers of Heteroscedastic neurons are: ' + str(X_sample_het))
        print('Best Number of Heteroscedastic Neurons is: ' + str(best_neurons_het))

        f_plot_interim = False

        if f_plot_interim:

            iter_x = np.arange(1, bayes_opt_iters + 1)
            ax = plt.gca()
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.plot(iter_x, obj_val_list_homo, label='Homoscedastic')
            plt.plot(iter_x, obj_val_list_het, label='Heteroscedastic')
            plt.title('Best Objective Function Value Found so Far')
            plt.xlabel('Iteration Number')
            plt.ylabel('Mean Classification Accuracy - One Std')
            plt.legend()
            plt.show()

        homo_running_sum += np.array(obj_val_list_homo)
        homo_squares += np.array(obj_val_list_homo)**2
        hetero_running_sum += np.array(obj_val_list_het)
        hetero_squares += np.array(obj_val_list_het)**2

        homo_noise_running_sum += np.array(noise_list_homo)
        homo_noise_squares += np.array(noise_list_homo)**2
        hetero_noise_running_sum += np.array(noise_list_het)
        hetero_noise_squares += np.array(noise_list_het)**2

    homo_means = homo_running_sum/random_trials
    hetero_means = hetero_running_sum/random_trials
    homo_errs = np.sqrt(homo_squares/random_trials - homo_means**2)
    hetero_errs = np.sqrt(hetero_squares/random_trials - hetero_means**2)

    print('List of average homoscedastic values is: ' + str(homo_means))
    print('List of homoscedastic errors is: ' + str(homo_errs))
    print('List of average heteroscedastic values is ' + str(hetero_means))
    print('List of heteroscedastic errors is: ' + str(hetero_errs))

    iter_x = np.arange(1, bayes_opt_iters + 1)

    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.plot(iter_x, homo_means, color='r', label='Homoscedastic')
    plt.plot(iter_x, hetero_means, color='b', label='Heteroscedastic')
    lower_homo = np.array(homo_means) - np.array(homo_errs)
    upper_homo = np.array(homo_means) + np.array(homo_errs)
    lower_hetero = np.array(hetero_means) - np.array(hetero_errs)
    upper_hetero = np.array(hetero_means) + np.array(hetero_errs)
    plt.fill_between(iter_x, lower_homo, upper_homo, color='r', label='Homoscedastic', alpha=0.1)
    plt.fill_between(iter_x, lower_hetero, upper_hetero, color='b', label='Heteroscedastic', alpha=0.1)
    plt.title('Best Objective Function Value Found so Far')
    plt.xlabel('Number of Function Evaluations')
    plt.ylabel('Mean Classification Accuracy - One Std')
    plt.legend(loc=4)
    plt.savefig('figures/stan_error_plot{}_and_{}_and_ntrials{}_and_numlayers{}'.format(bayes_opt_iters, bounds[1], n_trials, num_layers))

    homo_noise_means = homo_noise_running_sum/random_trials
    hetero_noise_means = hetero_noise_running_sum/random_trials
    homo_noise_errs = np.sqrt(homo_noise_squares/random_trials - homo_noise_means**2)
    hetero_noise_errs = np.sqrt(hetero_noise_squares/random_trials - hetero_noise_means**2)

    print('List of average homoscedastic noise values is: ' + str(homo_noise_means))
    print('List of homoscedastic noise errors is: ' + str(homo_noise_errs))
    print('List of average heteroscedastic noise values is ' + str(hetero_noise_means))
    print('List of heteroscedastic noise errors is: ' + str(hetero_noise_errs))

    plt.clf()
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.plot(iter_x, homo_noise_means, color='r', label='Homoscedastic')
    plt.plot(iter_x, hetero_noise_means, color='b', label='Heteroscedastic')
    lower_noise_homo = np.array(homo_noise_means) - np.array(homo_noise_errs)
    upper_noise_homo = np.array(homo_noise_means) + np.array(homo_noise_errs)
    lower_noise_hetero = np.array(hetero_noise_means) - np.array(hetero_noise_errs)
    upper_noise_hetero = np.array(hetero_noise_means) + np.array(hetero_noise_errs)
    plt.fill_between(iter_x, lower_noise_homo, upper_noise_homo, color='r', label='Homoscedastic', alpha=0.1)
    plt.fill_between(iter_x, lower_noise_hetero, upper_noise_hetero, color='b', label='Heteroscedastic', alpha=0.1)
    plt.title('Best Aleatoric Noise Found so Far')
    plt.xlabel('Number of Function Evaluations')
    plt.ylabel('Ground Truth Aleatoric Noise')
    plt.legend()
    plt.savefig('figures/stan_aleatoric_noise_plot{}_and_{}_and_ntrials{}_and_numlayers{}'.format(bayes_opt_iters, bounds[1], n_trials, num_layers))
