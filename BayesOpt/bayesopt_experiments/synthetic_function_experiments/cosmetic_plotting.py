"""
Script for cosmetic plotting of BayesOpt traces.
"""

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

if __name__ == '__main__':

    heteroscedastic = True
    opt_func = 'branin'  # One of ['branin', 'hosaki', 'goldstein']
    exp_type = 'hetero'  # One of ['homoscedastic', 'hetero', 'noiseless']
    bayes_opt_iters = 10
    iter_x = np.arange(1, bayes_opt_iters + 1)

    # Load data for cosmetic plotting

    rand_means = np.loadtxt(f'synth_saved_data/{exp_type}/{opt_func}/rand_means.txt')
    homo_means = np.loadtxt(f'synth_saved_data/{exp_type}/{opt_func}/homo_means.txt')
    hetero_means = np.loadtxt(f'synth_saved_data/{exp_type}/{opt_func}/hetero_means.txt')
    aug_means = np.loadtxt(f'synth_saved_data/{exp_type}/{opt_func}/aug_means.txt')
    aug_het_means = np.loadtxt(f'synth_saved_data/{exp_type}/{opt_func}/aug_het_means.txt')

    lower_rand = np.loadtxt(f'synth_saved_data/{exp_type}/{opt_func}/lower_rand.txt')
    upper_rand = np.loadtxt(f'synth_saved_data/{exp_type}/{opt_func}/upper_rand.txt')
    lower_homo = np.loadtxt(f'synth_saved_data/{exp_type}/{opt_func}/lower_homo.txt')
    upper_homo = np.loadtxt(f'synth_saved_data/{exp_type}/{opt_func}/upper_homo.txt')
    lower_hetero = np.loadtxt(f'synth_saved_data/{exp_type}/{opt_func}/lower_hetero.txt')
    upper_hetero = np.loadtxt(f'synth_saved_data/{exp_type}/{opt_func}/upper_hetero.txt')
    lower_aei = np.loadtxt(f'synth_saved_data/{exp_type}/{opt_func}/lower_aei.txt')
    upper_aei = np.loadtxt(f'synth_saved_data/{exp_type}/{opt_func}/upper_aei.txt')
    lower_het_aei = np.loadtxt(f'synth_saved_data/{exp_type}/{opt_func}/lower_het_aei.txt')
    upper_het_aei = np.loadtxt(f'synth_saved_data/{exp_type}/{opt_func}/upper_het_aei.txt')

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

    plt.xlabel('Function Evaluations', fontsize=14)
    if heteroscedastic:
        plt.ylabel('f(x) + g(x)', fontsize=14)
    else:
        plt.ylabel('f(x)', fontsize=14)
    plt.tick_params(labelsize=14)
    plt.legend(loc='lower left', bbox_to_anchor=(0.0, -0.425), ncol=3, borderaxespad=0, fontsize=14, frameon=False)
    #plt.yticks([4, 6, 8, 10])
    plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    if exp_type == 'homoscedastic':
        tag = 'with_noise_'
    else:
        if heteroscedastic:
            tag = 'heteroscedastic'
        else:
            tag = ''
    plt.savefig('cosmetic_figures/{}/bayesopt_plot{}_iters_{}'.format(opt_func, bayes_opt_iters, tag), bbox_inches='tight')

    plt.close()
    plt.cla()

    if heteroscedastic:
        rand_noise_means = np.loadtxt(f'synth_saved_data/{exp_type}/{opt_func}/rand_noise_means.txt')
        homo_noise_means = np.loadtxt(f'synth_saved_data/{exp_type}/{opt_func}/homo_noise_means.txt')
        hetero_noise_means = np.loadtxt(f'synth_saved_data/{exp_type}/{opt_func}/hetero_noise_means.txt')
        aug_noise_means = np.loadtxt(f'synth_saved_data/{exp_type}/{opt_func}/aug_noise_means.txt')
        aug_het_noise_means = np.loadtxt(f'synth_saved_data/{exp_type}/{opt_func}/aug_het_noise_means.txt')

        lower_noise_rand = np.loadtxt(f'synth_saved_data/{exp_type}/{opt_func}/lower_noise_rand.txt')
        upper_noise_rand = np.loadtxt(f'synth_saved_data/{exp_type}/{opt_func}/upper_noise_rand.txt')
        lower_noise_homo = np.loadtxt(f'synth_saved_data/{exp_type}/{opt_func}/lower_noise_homo.txt')
        upper_noise_homo = np.loadtxt(f'synth_saved_data/{exp_type}/{opt_func}/upper_noise_homo.txt')
        lower_noise_hetero = np.loadtxt(f'synth_saved_data/{exp_type}/{opt_func}/lower_noise_hetero.txt')
        upper_noise_hetero = np.loadtxt(f'synth_saved_data/{exp_type}/{opt_func}/upper_noise_hetero.txt')
        lower_noise_aei = np.loadtxt(f'synth_saved_data/{exp_type}/{opt_func}/lower_noise_aei.txt')
        upper_noise_aei = np.loadtxt(f'synth_saved_data/{exp_type}/{opt_func}/upper_noise_aei.txt')
        lower_noise_het_aei = np.loadtxt(f'synth_saved_data/{exp_type}/{opt_func}/lower_noise_het_aei.txt')
        upper_noise_het_aei = np.loadtxt(f'synth_saved_data/{exp_type}/{opt_func}/upper_noise_het_aei.txt')

        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

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
        plt.xlabel('Function Evaluations', fontsize=14)
        plt.ylabel('g(x)', fontsize=14)
        plt.tick_params(labelsize=14)
        plt.legend(loc='lower left', bbox_to_anchor=(0.0, -0.425), ncol=3, borderaxespad=0, fontsize=14, frameon=False)
        plt.savefig('cosmetic_figures/{}/heteroscedastic_bayesopt_plot{}_iters'
                    'noise_only'.format(opt_func, bayes_opt_iters), bbox_inches='tight')
