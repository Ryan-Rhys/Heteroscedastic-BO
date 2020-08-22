# Copyright Ryan-Rhys Griffiths 2019
# Author: Ryan-Rhys Griffiths
"""
This module contains the code for benchmarking heteroscedastic Bayesian Optimisation on a number of toy functions.
"""

import matplotlib.pyplot as plt
import numpy as np

from acquisition_functions import heteroscedastic_expected_improvement, heteroscedastic_propose_location, my_propose_location, my_expected_improvement
from objective_functions import branin_function, min_branin_noise_function, heteroscedastic_branin, linear_sin_noise, max_sin_noise_objective


if __name__ == '__main__':

    # We perform random trials of Bayesian Optimisation

    random_trials = 10

    for i in range(random_trials):

        numpy_seed = i + 50
        tf_seed = i + 51
        np.random.seed(numpy_seed)

        noise = 0.2
        bounds = np.array([[-5.0, 10.0], [0.0, 15.0]])  # bounds of the Bayesian Optimisation problem.

        #  Initial noisy data points sampled uniformly at random from the input space.

        grid_size = 3

        x1 = np.random.uniform(-5.0, 10.0, size=(grid_size,))
        x2 = np.random.uniform(0.0, 15.0, size=(grid_size,))
        X_init = np.array(np.meshgrid(x1, x2)).T.reshape(-1, 2)
        #Y_init = branin_function(X_init[:, 0], X_init[:, 1], noise)
        Y_init = heteroscedastic_branin(X_init[:, 0], X_init[:, 1])

        # Dense grid of points within bounds

        x1_star = np.arange(-5.0, 10.0, 0.5)
        x2_star = np.arange(0.0, 15.0, 0.5)
        plot_sample = np.array(np.meshgrid(x1_star, x2_star)).T.reshape(-1, 2)  # Where 2 gives the dimensionality

        # Initialize samples
        X_sample = X_init
        Y_sample = Y_init

        # Number of iterations
        n_iter = 20

        # initial BayesOpt hypers

        l_init = 1.0
        sigma_f_init = 1.0
        l_noise_init = 1
        sigma_f_noise_init = 1.0
        gp2_noise = 1.0
        num_iters = 10
        sample_size = 100

        best_so_far = 300  # value to beat
        obj_val_list = []
        collected_x1 = []
        collected_x2 = []

        for i in range(n_iter):

            print(i)

            # Obtain next sampling point from the acquisition function (expected_improvement)

            X_next = my_propose_location(my_expected_improvement, X_sample, Y_sample, noise, l_init, sigma_f_init, bounds,
                                         plot_sample, n_restarts=3, min_val=300)

            # X_next = heteroscedastic_propose_location(heteroscedastic_expected_improvement, X_sample, Y_sample, noise, l_init,
            #                                           sigma_f_init, l_noise_init, sigma_f_noise_init, gp2_noise, num_iters,
            #                                           sample_size, bounds, plot_sample, n_restarts=3, min_val=300)

            collected_x1.append(X_next[:, 0])
            collected_x2.append(X_next[:, 1])

            # Obtain next noisy sample from the objective function
            Y_next = heteroscedastic_branin(X_next[:, 0], X_next[:, 1])
            obj_val = min_branin_noise_function(X_next[:, 0], X_next[:, 1])[0]
            print(obj_val)

            if obj_val < best_so_far:
                best_so_far = obj_val
                obj_val_list.append(obj_val)
            else:
                obj_val_list.append(best_so_far)

            # Add sample to previous samples
            X_sample = np.vstack((X_sample, X_next))
            Y_sample = np.vstack((Y_sample, Y_next))

        plt.plot(np.arange(1, n_iter + 1, 1), np.array(obj_val_list))
        plt.xlabel('Iteration Number')
        plt.ylabel('Objective Function Value')
        plt.title('Best value obtained so far')
        plt.show()

        plt.plot(np.array(collected_x1), np.array(collected_x2), '+', color='green', markersize='12', linewidth='8')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title('Collected Data Points')
        plt.show()
