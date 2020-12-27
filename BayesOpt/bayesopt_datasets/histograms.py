# Author: Ryan-Rhys Griffiths
"""
Script to generate histogroms of the FreeSolv dataset.
"""

from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

from bayesopt_experiments.data_utils import parse_dataset

FREESOLV_PATH = '../bayesopt_datasets/Freesolv/Freesolv.txt'
task = 'FreeSolv'
use_frag = True
use_exp = True  # use experimental values.

if __name__ == '__main__':

    _, _, std_exp = parse_dataset(task, FREESOLV_PATH, use_frag, use_exp=True)
    _, _, std_calc = parse_dataset(task, FREESOLV_PATH, use_frag, use_exp=False)

    plt.hist(std_exp, color=(1, 0, 0, 0.5), bins=6, density=True)
    density = gaussian_kde(std_exp)
    xs = np.linspace(np.min(std_exp), np.max(std_exp), 200)
    plt.plot(xs, density(xs), color='k')
    plt.yticks([0, 1, 2, 3])
    plt.xticks([0, 0.5, 1, 1.5, 2])
    plt.xlabel('Error Magnitude (kcal/mol)', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.tick_params(labelsize=14)
    plt.savefig('Histograms/freesolv_exp.png')
    plt.close()

    plt.cla()
    plt.hist(std_calc, color=(0, 0, 1, 0.5), bins=6, density=True)
    density = gaussian_kde(std_calc)
    xs = np.linspace(np.min(std_calc), np.max(std_calc), 200)
    plt.plot(xs, density(xs), color='k')
    plt.xticks([0.02, 0.04, 0.06, 0.08])
    plt.xlabel('Error Magnitude (kcal/mol)', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.tick_params(labelsize=14)
    plt.savefig('Histograms/freesolv_calc.png')

