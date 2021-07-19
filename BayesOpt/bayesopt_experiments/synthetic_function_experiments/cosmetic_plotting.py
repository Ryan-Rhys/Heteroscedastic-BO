"""
Script for cosmetic plotting of BayesOpt traces.
"""

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

if __name__ == '__main__':

    heteroscedastic = True
    opt_func = 'hosaki'  # One of ['branin', 'hosaki', 'goldstein']
    exp_type = 'kernel'  # One of ['homoscedastic', 'hetero', 'noiseless', 'gamma', 'kernel']
    bayes_opt_iters = 10
    iter_x = np.arange(1, bayes_opt_iters + 1)

    # Load data for cosmetic plotting

    if exp_type == 'kernel':

        assert heteroscedastic == True

        hetero_means_rbf = np.loadtxt(f'synth_saved_data/hetero/{opt_func}/hetero_means.txt')
        lower_hetero_rbf = np.loadtxt(f'synth_saved_data/hetero/{opt_func}/lower_hetero.txt')
        upper_hetero_rbf = np.loadtxt(f'synth_saved_data/hetero/{opt_func}/upper_hetero.txt')

        hetero_means_mat12 = np.loadtxt(f'synth_saved_data/hetero/{opt_func}/hetero_means_mat12.txt')
        lower_hetero_mat12 = np.loadtxt(f'synth_saved_data/hetero/{opt_func}/lower_hetero_mat12.txt')
        upper_hetero_mat12 = np.loadtxt(f'synth_saved_data/hetero/{opt_func}/upper_hetero_mat12.txt')

        hetero_means_mat52 = np.loadtxt(f'synth_saved_data/hetero/{opt_func}/hetero_means_mat52.txt')
        lower_hetero_mat52 = np.loadtxt(f'synth_saved_data/hetero/{opt_func}/lower_hetero_mat52.txt')
        upper_hetero_mat52 = np.loadtxt(f'synth_saved_data/hetero/{opt_func}/upper_hetero_mat52.txt')

        plt.plot(iter_x, hetero_means_rbf, color='#006600', label=r'RBF')
        plt.fill_between(iter_x, lower_hetero_rbf, upper_hetero_rbf, color='#006600', alpha=0.1)
        plt.plot(iter_x, hetero_means_mat12, color='#003366', label=r'Matern 1/2')
        plt.fill_between(iter_x, lower_hetero_mat12, upper_hetero_mat12, color='#003366', alpha=0.1)
        plt.plot(iter_x, hetero_means_mat52, color='#99004C', label=r'Matern 5/2')
        plt.fill_between(iter_x, lower_hetero_mat52, upper_hetero_mat52, color='#99004C', alpha=0.1)
        plt.xlabel('Function Evaluations', fontsize=14)

        plt.ylabel('f(x) + g(x)', fontsize=14)
        plt.tick_params(labelsize=14)
        plt.legend(loc='lower left', bbox_to_anchor=(0.0, -0.425), ncol=3, borderaxespad=0, fontsize=14, frameon=False)
        # plt.yticks([4, 6, 8, 10])
        # plt.yticks([-1.5, -1, -0.5, 0])
        plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        tag = 'kernel_test_anpei'

        plt.savefig('cosmetic_figures/{}/bayesopt_plot{}_iters_{}_kernel'.format(opt_func, bayes_opt_iters, tag),
                    bbox_inches='tight')

        plt.close()
        plt.cla()

        aug_het_means_rbf = np.loadtxt(f'synth_saved_data/hetero/{opt_func}/aug_het_means.txt')
        lower_het_aei_rbf = np.loadtxt(f'synth_saved_data/hetero/{opt_func}/lower_het_aei.txt')
        upper_het_aei_rbf = np.loadtxt(f'synth_saved_data/hetero/{opt_func}/upper_het_aei.txt')

        aug_het_means_mat12 = np.loadtxt(f'synth_saved_data/hetero/{opt_func}/aug_het_means_mat12.txt')
        lower_het_aei_mat12 = np.loadtxt(f'synth_saved_data/hetero/{opt_func}/lower_het_aei_mat12.txt')
        upper_het_aei_mat12 = np.loadtxt(f'synth_saved_data/hetero/{opt_func}/upper_het_aei_mat12.txt')

        aug_het_means_mat52 = np.loadtxt(f'synth_saved_data/hetero/{opt_func}/aug_het_means_mat52.txt')
        lower_het_aei_mat52 = np.loadtxt(f'synth_saved_data/hetero/{opt_func}/lower_het_aei_mat52.txt')
        upper_het_aei_mat52 = np.loadtxt(f'synth_saved_data/hetero/{opt_func}/upper_het_aei_mat52.txt')

        # aug_het_means1000 = np.loadtxt(f'synth_saved_data/{exp_type}/aug_het_means_aleatoric_weight_is_500_aug.txt')
        # lower_het_aei1000 = np.loadtxt(f'synth_saved_data/{exp_type}/lower_het_aei_aleatoric_weight_is_500_aug.txt')
        # upper_het_aei1000 = np.loadtxt(f'synth_saved_data/{exp_type}/upper_het_aei_aleatoric_weight_is_500_aug.txt')

        plt.plot(iter_x, aug_het_means_rbf, color='#006600', label=r'RBF')
        plt.fill_between(iter_x, lower_het_aei_rbf, upper_het_aei_rbf, color='#006600', alpha=0.1)
        plt.plot(iter_x, aug_het_means_mat12, color='#003366', label=r'Matern 1/2')
        plt.fill_between(iter_x, lower_het_aei_mat12, upper_het_aei_mat12, color='#003366', alpha=0.1)
        plt.plot(iter_x, aug_het_means_mat52, color='#99004C', label=r'Matern 5/2')
        plt.fill_between(iter_x, lower_het_aei_mat52, upper_het_aei_mat52, color='#99004C', alpha=0.1)
        # plt.plot(iter_x, aug_het_means1000, color='#FF9933', label=r'$\gamma = 500$')
        # plt.fill_between(iter_x, lower_het_aei1000, upper_het_aei1000, color='#FF9933', alpha=0.1)
        plt.xlabel('Function Evaluations', fontsize=14)
        if heteroscedastic:
            plt.ylabel('f(x) + g(x)', fontsize=14)
        else:
            plt.ylabel('f(x)', fontsize=14)
        plt.tick_params(labelsize=14)
        plt.legend(loc='lower left', bbox_to_anchor=(0.0, -0.425), ncol=3, borderaxespad=0, fontsize=14, frameon=False)
        plt.yticks([6, 7, 8, 9])
        # plt.yticks([-1.5, -1, -0.5, 0])
        plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        tag = 'kernel_test_haei'

        plt.savefig('cosmetic_figures/{}/bayesopt_plot{}_iters_{}_kernel'.format(opt_func, bayes_opt_iters, tag),
                    bbox_inches='tight')

    elif exp_type == 'gamma':

        assert heteroscedastic == True

        hetero_means_1 = np.loadtxt(f'synth_saved_data/{exp_type}/hetero_means_aleatoric_weight_is_1.txt')
        lower_hetero_1 = np.loadtxt(f'synth_saved_data/{exp_type}/lower_hetero_aleatoric_weight_is_1.txt')
        upper_hetero_1 = np.loadtxt(f'synth_saved_data/{exp_type}/upper_hetero_aleatoric_weight_is_1.txt')

        hetero_means_10 = np.loadtxt(f'synth_saved_data/{exp_type}/hetero_means_aleatoric_weight_is_10.txt')
        lower_hetero_10 = np.loadtxt(f'synth_saved_data/{exp_type}/lower_hetero_aleatoric_weight_is_10.txt')
        upper_hetero_10 = np.loadtxt(f'synth_saved_data/{exp_type}/upper_hetero_aleatoric_weight_is_10.txt')

        hetero_means_100 = np.loadtxt(f'synth_saved_data/{exp_type}/hetero_means_aleatoric_weight_is_100.txt')
        lower_hetero_100 = np.loadtxt(f'synth_saved_data/{exp_type}/lower_hetero_aleatoric_weight_is_100.txt')
        upper_hetero_100 = np.loadtxt(f'synth_saved_data/{exp_type}/upper_hetero_aleatoric_weight_is_100.txt')

        plt.plot(iter_x, hetero_means_1, color='#006600', label=r'$\beta = 0.5$')
        plt.fill_between(iter_x, lower_hetero_1, upper_hetero_1, color='#006600', alpha=0.1)
        plt.plot(iter_x, hetero_means_10, color='#003366', label=r'$\beta = 1/11$')
        plt.fill_between(iter_x, lower_hetero_10, upper_hetero_10, color='#003366', alpha=0.1)
        plt.plot(iter_x, hetero_means_100, color='#99004C', label=r'$\beta = 1/101$')
        plt.fill_between(iter_x, lower_hetero_100, upper_hetero_100, color='#99004C', alpha=0.1)
        plt.xlabel('Function Evaluations', fontsize=14)

        plt.ylabel('f(x) + g(x)', fontsize=14)
        plt.tick_params(labelsize=14)
        plt.legend(loc='lower left', bbox_to_anchor=(0.0, -0.425), ncol=3, borderaxespad=0, fontsize=14, frameon=False)
        # plt.yticks([4, 6, 8, 10])
        # plt.yticks([-1.5, -1, -0.5, 0])
        plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        tag = 'anpei'

        plt.savefig('cosmetic_figures/{}/bayesopt_plot{}_iters_{}_weight'.format(opt_func, bayes_opt_iters, tag),
                    bbox_inches='tight')

        plt.close()
        plt.cla()

        aug_het_means1 = np.loadtxt(f'synth_saved_data/{exp_type}/aug_het_means_aleatoric_weight_is_1.txt')
        lower_het_aei1 = np.loadtxt(f'synth_saved_data/{exp_type}/lower_het_aei_aleatoric_weight_is_1.txt')
        upper_het_aei1 = np.loadtxt(f'synth_saved_data/{exp_type}/upper_het_aei_aleatoric_weight_is_1.txt')

        aug_het_means10 = np.loadtxt(f'synth_saved_data/{exp_type}/aug_het_means_aleatoric_weight_is_10.txt')
        lower_het_aei10 = np.loadtxt(f'synth_saved_data/{exp_type}/lower_het_aei_aleatoric_weight_is_10.txt')
        upper_het_aei10 = np.loadtxt(f'synth_saved_data/{exp_type}/upper_het_aei_aleatoric_weight_is_10.txt')

        aug_het_means100 = np.loadtxt(f'synth_saved_data/{exp_type}/aug_het_means_aleatoric_weight_is_100.txt')
        lower_het_aei100 = np.loadtxt(f'synth_saved_data/{exp_type}/lower_het_aei_aleatoric_weight_is_100.txt')
        upper_het_aei100 = np.loadtxt(f'synth_saved_data/{exp_type}/upper_het_aei_aleatoric_weight_is_100.txt')

        # aug_het_means1000 = np.loadtxt(f'synth_saved_data/{exp_type}/aug_het_means_aleatoric_weight_is_500_aug.txt')
        # lower_het_aei1000 = np.loadtxt(f'synth_saved_data/{exp_type}/lower_het_aei_aleatoric_weight_is_500_aug.txt')
        # upper_het_aei1000 = np.loadtxt(f'synth_saved_data/{exp_type}/upper_het_aei_aleatoric_weight_is_500_aug.txt')

        plt.plot(iter_x, aug_het_means1, color='#006600', label=r'$\gamma = 1$')
        plt.fill_between(iter_x, lower_het_aei1, upper_het_aei1, color='#006600', alpha=0.1)
        plt.plot(iter_x, aug_het_means10, color='#003366', label=r'$\gamma = 10$')
        plt.fill_between(iter_x, lower_het_aei10, upper_het_aei10, color='#003366', alpha=0.1)
        plt.plot(iter_x, aug_het_means100, color='#99004C', label=r'$\gamma = 100$')
        plt.fill_between(iter_x, lower_het_aei100, upper_het_aei100, color='#99004C', alpha=0.1)
        # plt.plot(iter_x, aug_het_means1000, color='#FF9933', label=r'$\gamma = 500$')
        # plt.fill_between(iter_x, lower_het_aei1000, upper_het_aei1000, color='#FF9933', alpha=0.1)
        plt.xlabel('Function Evaluations', fontsize=14)
        if heteroscedastic:
            plt.ylabel('f(x) + g(x)', fontsize=14)
        else:
            plt.ylabel('f(x)', fontsize=14)
        plt.tick_params(labelsize=14)
        plt.legend(loc='lower left', bbox_to_anchor=(0.0, -0.425), ncol=3, borderaxespad=0, fontsize=14, frameon=False)
        plt.yticks([6, 7, 8, 9])
        # plt.yticks([-1.5, -1, -0.5, 0])
        plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        tag = 'haei'

        plt.savefig('cosmetic_figures/{}/bayesopt_plot{}_iters_{}_weight'.format(opt_func, bayes_opt_iters, tag),
                    bbox_inches='tight')

    else:

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
        #plt.yticks([-1.5, -1, -0.5, 0])
        plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        if exp_type == 'homoscedastic':
            tag = 'with_noise_'
        elif exp_type == 'noiseless':
            tag = 'noiseless_'
        else:
            if heteroscedastic:
                tag = 'heteroscedastic'
            else:
                tag = ''
        plt.savefig('cosmetic_figures/{}/bayesopt_plot{}_iters_{}3'.format(opt_func, bayes_opt_iters, tag), bbox_inches='tight')

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
            lower_noise_aei = np.loadtxt(f'synth_saved_data/{exp_type}/{opt_func}/lower_noise_aei_mat32.txt')
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
                        'noise_only3'.format(opt_func, bayes_opt_iters), bbox_inches='tight')
