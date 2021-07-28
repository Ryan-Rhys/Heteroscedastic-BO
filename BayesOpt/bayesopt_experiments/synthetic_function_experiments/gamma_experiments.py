# Copyright Ryan-Rhys Griffiths 2021
# Author: Ryan-Rhys Griffiths
"""
Script for describing effect of the gamma parameter of HAEI and ANPEI on synthetic functions with heteroscedastic noise.
"""

import argparse
import sys

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

sys.path.append('../')
sys.path.append('../..')
sys.path.append('../../..')

from acquisition_funcs.acquisition_functions import heteroscedastic_expected_improvement, heteroscedastic_propose_location, \
    heteroscedastic_augmented_expected_improvement
from BayesOpt.objective_funcs.synthetic_functions import hosaki_function, branin_function, goldstein_price_function


def main(penalty, aleatoric_weight, aleatoric_weight_aug, random_trials, bayes_opt_iters, grid_size, exp_type, opt_func,
         noise_level):
    """
    Optimise the heteroscedastic Branin-Hoo function.

    param: penalty: $\alpha$ parameter specifying weight of noise component to objective
    param: aleatoric_weight: float specifying the value of $\beta of ANPEI
    param: aleatoric_weight_aug: float specifying the value of $\gamma of HAEI
    param: random_trials: int specifying the number of random initialisations
    param: bayes_opt_iters: int specifying the number of iterations of BayesOpt
    param: grid_size: int specifying the side length of the 2D grid to initialise on.
    param: exp_type: str specifying the type of experiment. One of ['hetero', 'homoscedastic', 'noiseless']
    param: opt_func: str specifying the optimisation function. One of ['hosaki', 'branin', 'goldstein']
    param: noise_level: int specifying the noise level for homoscedastic noise. Should be 0 when heteroscedastic.
    """

    if noise_level != 0:
        assert exp_type == 'homoscedastic'
    if exp_type == 'hetero':
        assert noise_level == 0
        heteroscedastic = True
    if heteroscedastic is not True and noise_level == 0:
        assert exp_type == 'noiseless'
    n_restarts = 20

    # We perform random trials of Bayesian Optimisation

    hetero_running_sum = np.zeros(bayes_opt_iters)
    hetero_squares = np.zeros(bayes_opt_iters)
    aug_het_running_sum = np.zeros(bayes_opt_iters)
    aug_het_squares = np.zeros(bayes_opt_iters)

    # We compute the objective corresponding to aleatoric noise only

    hetero_noise_running_sum = np.zeros(bayes_opt_iters)
    hetero_noise_squares = np.zeros(bayes_opt_iters)
    aug_het_noise_running_sum = np.zeros(bayes_opt_iters)
    aug_het_noise_squares = np.zeros(bayes_opt_iters)

    for i in range(random_trials):

        numpy_seed = i
        np.random.seed(numpy_seed)

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

        het_best_so_far = -300
        aug_het_best_so_far = -300
        het_noise_best_so_far = 300
        aug_het_noise_best_so_far = 300
        het_obj_val_list = []
        aug_het_obj_val_list = []
        het_noise_val_list = []
        aug_het_noise_val_list = []
        het_collected_x1 = []
        het_collected_x2 = []
        aug_het_collected_x1 = []
        aug_het_collected_x2 = []

        for j in range(bayes_opt_iters):

            print(j)

            # random sampling baseline

            seed = bayes_opt_iters*i + j  # This approach
            print(f'Seed is: {seed}')
            np.random.seed(seed)

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

            # Obtain next sampling point from the heteroscedastic augmented expected improvement (het-AEI)

            aug_het_X_next = heteroscedastic_propose_location(heteroscedastic_augmented_expected_improvement, aug_het_X_sample,
                                                          aug_het_Y_sample, noise, l_init, sigma_f_init, l_noise_init,
                                                          sigma_f_noise_init, gp2_noise, num_iters, sample_size, bounds,
                                                          plot_sample, n_restarts=n_restarts, min_val=300, aleatoric_weight=aleatoric_weight_aug)

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

            if aug_het_noise_val < aug_het_noise_best_so_far:
                aug_het_noise_best_so_far = aug_het_noise_val
                aug_het_noise_val_list.append(aug_het_noise_val)
            else:
                aug_het_noise_val_list.append(aug_het_noise_best_so_far)

            # Add sample to previous sample
            aug_het_X_sample = np.vstack((aug_het_X_sample, aug_het_X_next))
            aug_het_Y_sample = np.vstack((aug_het_Y_sample, aug_het_Y_next))

        hetero_running_sum += np.array(het_obj_val_list, dtype=np.float64).flatten()
        hetero_squares += np.array(het_obj_val_list, dtype=np.float64).flatten() ** 2
        aug_het_running_sum += np.array(aug_het_obj_val_list, dtype=np.float64).flatten()
        aug_het_squares += np.array(aug_het_obj_val_list, dtype=np.float64).flatten() ** 2

        hetero_noise_running_sum += np.array(het_noise_val_list, dtype=np.float64).flatten()
        hetero_noise_squares += np.array(het_noise_val_list, dtype=np.float64).flatten() ** 2
        aug_het_noise_running_sum += np.array(aug_het_noise_val_list, dtype=np.float64).flatten()
        aug_het_noise_squares += np.array(aug_het_noise_val_list, dtype=np.float64).flatten() ** 2

    # results are negated to turn problem into minimisation for consistency.
    hetero_means = -hetero_running_sum / random_trials
    hetero_errs = (np.sqrt(hetero_squares / random_trials - hetero_means ** 2, dtype=np.float64))/np.sqrt(random_trials)
    aug_het_means = -aug_het_running_sum / random_trials
    aug_het_errs = (np.sqrt(aug_het_squares / random_trials - aug_het_means **2, dtype=np.float64))/np.sqrt(random_trials)

    hetero_noise_means = hetero_noise_running_sum / random_trials
    hetero_noise_errs = (np.sqrt(hetero_noise_squares / random_trials - hetero_noise_means ** 2))/np.sqrt(random_trials)
    aug_het_noise_means = aug_het_noise_running_sum / random_trials
    aug_het_noise_errs = (np.sqrt(aug_het_noise_squares / random_trials - aug_het_noise_means ** 2))/np.sqrt(random_trials)

    print('List of average heteroscedastic values is ' + str(hetero_means))
    print('List of heteroscedastic errors is: ' + str(hetero_errs))
    print('List of average het-AEI values is: ' + str(aug_het_means))
    print('List of het-AEI errors is: ' + str(aug_het_errs))

    iter_x = np.arange(1, bayes_opt_iters + 1)

    # clear figure from previous fplot
    plt.cla()

    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    lower_hetero = np.array(hetero_means) - np.array(hetero_errs)
    upper_hetero = np.array(hetero_means) + np.array(hetero_errs)
    lower_het_aei = np.array(aug_het_means) - np.array(aug_het_errs)
    upper_het_aei = np.array(aug_het_means) + np.array(aug_het_errs)

    plt.plot(iter_x, hetero_means, color='tab:green', label='ANPEI')
    plt.plot(iter_x, aug_het_means, color='tab:purple', label='HAEI')
    plt.fill_between(iter_x, lower_hetero, upper_hetero, color='tab:green', alpha=0.1)
    plt.fill_between(iter_x, lower_het_aei, upper_het_aei, color='tab:purple', alpha=0.1)

    plt.title('Best Objective Function Value Found so Far', fontsize=16)
    plt.xlabel('Function Evaluations', fontsize=14)
    if penalty != 1:
        plt.ylabel('f(x) +' + str(penalty) + 'g(x)', fontsize=14)
    else:
        plt.ylabel('f(x) + g(x)', fontsize=14)
    plt.tick_params(labelsize=14)
    plt.legend(loc='lower left', bbox_to_anchor=(0.0, -0.425), ncol=3, borderaxespad=0, fontsize=14, frameon=False)
    tag = 'heteroscedastic'
    plt.savefig('gamma_figures/{}_{}_iters_{}_random_trials_and_grid_size_of_{}_and_seed_{}'
                '_hundred_times_penalty_is_{}_aleatoric_weight_aug_is_{}_{}_aug'.
                format(opt_func, bayes_opt_iters, random_trials, grid_size, numpy_seed, int(100*penalty), aleatoric_weight_aug, tag), bbox_inches='tight')

    plt.close()

    # clear figure from previous fplot returns if fiddling with form of function
    plt.cla()

    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    lower_noise_hetero = np.array(hetero_noise_means) - np.array(hetero_noise_errs)
    upper_noise_hetero = np.array(hetero_noise_means) + np.array(hetero_noise_errs)
    lower_noise_het_aei = np.array(aug_het_noise_means) - np.array(aug_het_noise_errs)
    upper_noise_het_aei = np.array(aug_het_noise_means) + np.array(aug_het_noise_errs)

    plt.plot(iter_x, hetero_noise_means, color='tab:green', label='ANPEI')
    plt.plot(iter_x, aug_het_noise_means, color='tab:purple', label='HAEI')
    plt.fill_between(iter_x, lower_noise_hetero, upper_noise_hetero, color='tab:green', alpha=0.1)
    plt.fill_between(iter_x, lower_noise_het_aei, upper_noise_het_aei, color='tab:purple', alpha=0.1)

    plt.title('Lowest Aleatoric Noise Found so Far', fontsize=16)
    plt.xlabel('Function Evaluations', fontsize=14)
    plt.ylabel('g(x)', fontsize=14)
    plt.tick_params(labelsize=14)
    plt.legend(loc='lower left', bbox_to_anchor=(0.0, -0.425), ncol=3, borderaxespad=0, fontsize=14, frameon=False)
    plt.savefig('gamma_figures/{}_{}_iters_{}_random_trials_and_grid_size_of_{}_and_seed_{}_'
                'noise_only_hundred_times_penalty_is_{}_aleatoric_weight_aug_is_{}_aug'.
                format(opt_func, bayes_opt_iters, random_trials, grid_size, numpy_seed, int(100*penalty), aleatoric_weight_aug), bbox_inches='tight')

    # Save data for cosmetic plotting

    np.savetxt(f'synth_saved_data/gamma/hetero_means_aleatoric_weight_is_{aleatoric_weight_aug}_aug.txt', hetero_means)
    np.savetxt(f'synth_saved_data/gamma/aug_het_means_aleatoric_weight_is_{aleatoric_weight_aug}_aug.txt', aug_het_means)

    np.savetxt(f'synth_saved_data/gamma/lower_hetero_aleatoric_weight_is_{aleatoric_weight_aug}_aug.txt', lower_hetero)
    np.savetxt(f'synth_saved_data/gamma/upper_hetero_aleatoric_weight_is_{aleatoric_weight_aug}_aug.txt', upper_hetero)
    np.savetxt(f'synth_saved_data/gamma/lower_het_aei_aleatoric_weight_is_{aleatoric_weight_aug}_aug.txt', lower_het_aei)
    np.savetxt(f'synth_saved_data/gamma/upper_het_aei_aleatoric_weight_is_{aleatoric_weight_aug}_aug.txt', upper_het_aei)

    np.savetxt(f'synth_saved_data/gamma/hetero_noise_means_aleatoric_weight_is_{aleatoric_weight_aug}_aug.txt', hetero_noise_means)
    np.savetxt(f'synth_saved_data/gamma/aug_het_noise_means_aleatoric_weight_is_{aleatoric_weight_aug}_aug.txt', aug_het_noise_means)

    np.savetxt(f'synth_saved_data/gamma/lower_noise_hetero_aleatoric_weight_is_{aleatoric_weight_aug}_aug.txt', lower_noise_hetero)
    np.savetxt(f'synth_saved_data/gamma/upper_noise_hetero_aleatoric_weight_is_{aleatoric_weight_aug}_aug.txt', upper_noise_hetero)
    np.savetxt(f'synth_saved_data/gamma/lower_noise_het_aei_aleatoric_weight_is_{aleatoric_weight_aug}_aug.txt', lower_noise_het_aei)
    np.savetxt(f'synth_saved_data/gamma/upper_noise_het_aei_aleatoric_weight_is_{aleatoric_weight_aug}_aug.txt', upper_noise_het_aei)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--penalty', type=int, default=1,
                        help='$\alpha$ parameter specifying weight of noise component to objective.')
    parser.add_argument('-a', '--aleatoric_weight', type=float, default=300,
                        help='The value of both $\beta of ANPEI')
    parser.add_argument('-ga', '--aleatoric_weight_aug', type=float, default=500,
                        help='The value of $\gamma of HAEI')
    parser.add_argument('-r', '--random_trials', type=int, default=50,
                        help='Number of random initialisations')
    parser.add_argument('-b', '--bayes_opt_iters', type=int, default=10,
                        help='The number of iterations of BayesOpt')
    parser.add_argument('-g', '--grid_size', type=int, default=6,
                        help='The grid size to intialise with i.e. the side length of a 2x2 grid')
    parser.add_argument('-e', '--exp_type', type=str, default='hetero',
                        help='The type of noise to use. One of [hetero, homoscedastic, noiseless]')
    parser.add_argument('-o', '--opt_func', type=str, default='branin',
                        help='The optimisation function to use. One of [branin, hosaki, goldstein]')
    parser.add_argument('-n', '--noise_level', type=int, default=0,
                        help='The noise level to use for homoscedatic noise experiments')

    args = parser.parse_args()

    main(args.penalty, args.aleatoric_weight, args.aleatoric_weight_aug, args.random_trials, args.bayes_opt_iters,
         args.grid_size, args.exp_type, args.opt_func, args.noise_level)
