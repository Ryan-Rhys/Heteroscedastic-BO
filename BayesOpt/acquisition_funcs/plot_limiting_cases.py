# Author: Ryan-Rhys Griffiths
"""
Script for plotting HAEI limiting cases.
"""

from matplotlib import pyplot as plt
import numpy as np


def haei_with_gamma(x, gamma):
    """
    compute HAEI acquisition with parameter gamma.
    """

    return 1 - (gamma/np.sqrt(gamma**2 + x))


def lin_approx(x, gamma):
    """
    Linear approximation for HAEI for small k.
    """

    return x/(2*gamma**2)


if __name__ == '__main__':
    x = np.linspace(0, 5, 100)
    gamma_list = [0.1, 1, 10]
    colours = [(1, 0, 0, 0.5), (0, 0, 1, 0.5), (0, 0.5, 0, 0.5)]
    i = 0
    for gamma in gamma_list:
        plt.plot(x, haei_with_gamma(x, gamma), label=f'$\gamma = {str(gamma)}$', color=colours[i], linewidth=4)
        if gamma != 0.1 and gamma != 1:
            plt.plot(x, lin_approx(x, gamma), '--', label=f'approx.', color='k')
        i += 1
    plt.xlabel('$k$', fontsize=14)
    plt.ylabel('$S(k)$', fontsize=14)
    plt.legend(loc=7, fontsize=12, bbox_to_anchor=(0.5, 0., 0.5, 0.55))
    plt.tick_params(labelsize=14)
    plt.savefig('figures/scaling_plot.png')
