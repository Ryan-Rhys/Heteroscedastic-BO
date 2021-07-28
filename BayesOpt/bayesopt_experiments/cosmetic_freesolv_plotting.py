"""
Script for plotting cosmetic freesolv results. Results are collected in seed batches due to Segfault issues when
running the Bayesian optimisation FreeSolv script on MacBook pro.
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

if __name__ == '__main__':

    fill = True
    numpy_seed = 0
    n_components = 14
    init_num_samples = 129
    penalty = 1
    aleatoric_weight = 1
    random_trials = 50
    start_seed = 0
    bayes_opt_iters = 10

    dir = '02_pen_1'  # One of ['02', '02_pen_1'] '02' has a penalty of 100.

    seed_indices = [8, 10, 12, 19, 47, 50]

    for seed_index in seed_indices:

        if seed_indices.index(seed_index) == 0:

            rand_running_sum = np.loadtxt(f'freesolv_data/{dir}/rand_means/rand_means_{start_seed}_{seed_index}.txt')
            rand_squares = np.loadtxt(f'freesolv_data/{dir}/rand_means/rand_squares_{start_seed}_{seed_index}.txt')
            homo_running_sum = np.loadtxt(f'freesolv_data/{dir}/homo_means/homo_means_{start_seed}_{seed_index}.txt')
            homo_squares = np.loadtxt(f'freesolv_data/{dir}/homo_means/homo_squares_{start_seed}_{seed_index}.txt')
            hetero_running_sum = np.loadtxt(f'freesolv_data/{dir}/het_means/hetero_means_{start_seed}_{seed_index}.txt')
            hetero_squares = np.loadtxt(f'freesolv_data/{dir}/het_means/hetero_squares_{start_seed}_{seed_index}.txt')
            aug_running_sum = np.loadtxt(f'freesolv_data/{dir}/aug_means/aug_means_{start_seed}_{seed_index}.txt')
            aug_squares = np.loadtxt(f'freesolv_data/{dir}/aug_means/aug_squares_{start_seed}_{seed_index}.txt')
            aug_het_running_sum = np.loadtxt(f'freesolv_data/{dir}/aug_het_means/aug_het_means_{start_seed}_{seed_index}.txt')
            aug_het_squares = np.loadtxt(f'freesolv_data/{dir}/aug_het_means/aug_het_squares_{start_seed}_{seed_index}.txt')

            rand_noise_running_sum = np.loadtxt(f'freesolv_data/{dir}/rand_noise/rand_means_{start_seed}_{seed_index}.txt')
            rand_noise_squares = np.loadtxt(f'freesolv_data/{dir}/rand_noise/rand_squares_{start_seed}_{seed_index}.txt')
            homo_noise_running_sum = np.loadtxt(f'freesolv_data/{dir}/homo_noise/homo_means_{start_seed}_{seed_index}.txt')
            homo_noise_squares = np.loadtxt(f'freesolv_data/{dir}/homo_noise/homo_squares_{start_seed}_{seed_index}.txt')
            hetero_noise_running_sum = np.loadtxt(f'freesolv_data/{dir}/het_noise/hetero_means_{start_seed}_{seed_index}.txt')
            hetero_noise_squares = np.loadtxt(f'freesolv_data/{dir}/het_noise/hetero_squares_{start_seed}_{seed_index}.txt')
            aug_noise_running_sum = np.loadtxt(f'freesolv_data/{dir}/aug_noise/aug_means_{start_seed}_{seed_index}.txt')
            aug_noise_squares = np.loadtxt(f'freesolv_data/{dir}/aug_noise/aug_squares_{start_seed}_{seed_index}.txt')
            aug_het_noise_running_sum = np.loadtxt(f'freesolv_data/{dir}/aug_het_noise/aug_het_means_{start_seed}_{seed_index}.txt')
            aug_het_noise_squares = np.loadtxt(f'freesolv_data/{dir}/aug_het_noise/aug_het_squares_{start_seed}_{seed_index}.txt')

        else:

            start_seed = seed_indices[seed_indices.index(seed_index) - 1]

            rand_running_sum += np.loadtxt(f'freesolv_data/{dir}/rand_means/rand_means_{start_seed}_{seed_index}.txt')
            rand_squares += np.loadtxt(f'freesolv_data/{dir}/rand_means/rand_squares_{start_seed}_{seed_index}.txt')
            homo_running_sum += np.loadtxt(f'freesolv_data/{dir}/homo_means/homo_means_{start_seed}_{seed_index}.txt')
            homo_squares += np.loadtxt(f'freesolv_data/{dir}/homo_means/homo_squares_{start_seed}_{seed_index}.txt')
            hetero_running_sum += np.loadtxt(f'freesolv_data/{dir}/het_means/hetero_means_{start_seed}_{seed_index}.txt')
            hetero_squares += np.loadtxt(f'freesolv_data/{dir}/het_means/hetero_squares_{start_seed}_{seed_index}.txt')
            aug_running_sum += np.loadtxt(f'freesolv_data/{dir}/aug_means/aug_means_{start_seed}_{seed_index}.txt')
            aug_squares += np.loadtxt(f'freesolv_data/{dir}/aug_means/aug_squares_{start_seed}_{seed_index}.txt')
            aug_het_running_sum += np.loadtxt(f'freesolv_data/{dir}/aug_het_means/aug_het_means_{start_seed}_{seed_index}.txt')
            aug_het_squares += np.loadtxt(f'freesolv_data/{dir}/aug_het_means/aug_het_squares_{start_seed}_{seed_index}.txt')

            rand_noise_running_sum += np.loadtxt(f'freesolv_data/{dir}/rand_noise/rand_means_{start_seed}_{seed_index}.txt')
            rand_noise_squares += np.loadtxt(f'freesolv_data/{dir}/rand_noise/rand_squares_{start_seed}_{seed_index}.txt')
            homo_noise_running_sum += np.loadtxt(f'freesolv_data/{dir}/homo_noise/homo_means_{start_seed}_{seed_index}.txt')
            homo_noise_squares += np.loadtxt(f'freesolv_data/{dir}/homo_noise/homo_squares_{start_seed}_{seed_index}.txt')
            hetero_noise_running_sum += np.loadtxt(f'freesolv_data/{dir}/het_noise/hetero_means_{start_seed}_{seed_index}.txt')
            hetero_noise_squares += np.loadtxt(f'freesolv_data/{dir}/het_noise/hetero_squares_{start_seed}_{seed_index}.txt')
            aug_noise_running_sum += np.loadtxt(f'freesolv_data/{dir}/aug_noise/aug_means_{start_seed}_{seed_index}.txt')
            aug_noise_squares += np.loadtxt(f'freesolv_data/{dir}/aug_noise/aug_squares_{start_seed}_{seed_index}.txt')
            aug_het_noise_running_sum += np.loadtxt(f'freesolv_data/{dir}/aug_het_noise/aug_het_means_{start_seed}_{seed_index}.txt')
            aug_het_noise_squares += np.loadtxt(f'freesolv_data/{dir}/aug_het_noise/aug_het_squares_{start_seed}_{seed_index}.txt')

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

    #plt.title('Best Objective Function Value Found so Far', fontsize=16)
    plt.xlabel('Function Evaluations', fontsize=14)
    if penalty != 1:
        plt.ylabel(f'Hydration Free Energy (kcal/mol) + {penalty}*Noise', fontsize=14)
    else:
        plt.ylabel(f'Hydration Free Energy (kcal/mol) + Noise', fontsize=14)
    plt.tick_params(labelsize=14)
    #plt.legend(loc=1)
    plt.legend(loc='lower left', bbox_to_anchor=(0.0, -0.425), ncol=3, borderaxespad=0, fontsize=14, frameon=False)
    plt.savefig('cosmetic_freesolv_figures/bayesopt_plot{}_iters_{}_random_trials_and_init_num_samples_of_{}_and_seed_{}_'
                'new_acq_penalty_is_{}_aleatoric_weight_is_{}_n_components_is_{}_new_aei_comp_seed_check_{}'.
                format(bayes_opt_iters, random_trials, init_num_samples, numpy_seed, penalty, aleatoric_weight, n_components, dir), bbox_inches='tight')

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

    #plt.title('Lowest Aleatoric Noise Found so Far', fontsize=16)
    plt.xlabel('Function Evaluations', fontsize=14)
    plt.ylabel('Aleatoric Noise', fontsize=14)
    plt.tick_params(labelsize=14)
    #plt.legend(loc=1)
    plt.legend(loc='lower left', bbox_to_anchor=(0.0, -0.425), ncol=3, borderaxespad=0, fontsize=14, frameon=False)
    plt.savefig('cosmetic_freesolv_figures/bayesopt_plot{}_iters_{}_random_trials_and_init_num_samples_of_{}_and_seed_{}_'
                'noise_only_new_acq_penalty_is_{}_aleatoric_weight_is_{}_n_components_is_{}_new_aei_comp_seed_check_{}'.
                format(bayes_opt_iters, random_trials, init_num_samples, numpy_seed, penalty, aleatoric_weight, n_components, dir), bbox_inches='tight')