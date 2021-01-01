# Copyright Lee Group 2019
# Author: Ryan-Rhys Griffiths
"""
This script contains utility data loading functions for the soil phosphorous fraction dataset.
"""

from matplotlib import pyplot as plt
import numpy as np


def soil(fplot_data=False):
    """
    Constructs the soil dataset.
    :param fplot_data: Boolean indicating whether or not to plot the dataset
    :return: x, y and errors
    """

    x_soil = []
    y_soil = []
    std_soil = []

    with open('../../bayesopt_datasets/Soil/soil_x.txt', 'r') as file:
        for line in file:
            x_data = line.split()
            x_soil.append(float(x_data[0]))
    with open('../../bayesopt_datasets/Soil/soil_y.txt', 'r') as file:
        for line in file:
            y_data = line.split()
            y_soil.append(float(y_data[0]))
    with open('../../bayesopt_datasets/Soil/soil_std.txt', 'r') as file:
        for line in file:
            std_data = line.split()
            std_soil.append(float(std_data[0]))

    x_soil = np.array(x_soil)
    y_soil = np.array(y_soil)
    std_soil = np.array(std_soil)

    if fplot_data:
        plt.plot(x_soil, y_soil, '+', color='green', markersize='12', linewidth='8')
        plt.xlabel('Density, dry bulk (g/cm^3)', fontsize=13)
        plt.ylabel('Inorganic Phosphorus (mg/kg)', fontsize=13)
        #plt.title('Soil Phosphorus Fraction as a Function of Density')
        plt.xticks([0, 0.5, 1, 1.5])
        plt.yticks([0, 150, 300])
        plt.savefig('figures/soil_dataset')
        plt.tick_params(labelsize=12)
        plt.close()

    return x_soil.reshape(-1, 1), y_soil.reshape(-1, 1), std_soil.reshape(-1, 1)


def soil_bo(fplot_data=False):
    """
    Constructs the soil dataset.
    :param fplot_data: Boolean indicating whether or not to plot the dataset
    :return: x, y and errors
    """

    x_soil = []
    y_soil = []
    std_soil = []

    import os

    with open('../bayesopt_datasets/Soil/soil_x.txt', 'r') as file:
        for line in file:
            x_data = line.split()
            x_soil.append(float(x_data[0]))
    with open('../bayesopt_datasets/Soil/soil_y.txt', 'r') as file:
        for line in file:
            y_data = line.split()
            y_soil.append(float(y_data[0]))
    with open('../bayesopt_datasets/Soil/soil_std.txt', 'r') as file:
        for line in file:
            std_data = line.split()
            std_soil.append(float(std_data[0]))

    x_soil = np.array(x_soil)
    y_soil = np.array(y_soil)
    std_soil = np.array(std_soil)

    if fplot_data:
        plt.plot(x_soil, y_soil, '+', color='green', markersize='12', linewidth='8')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Soil')
        plt.show()

    return x_soil.reshape(-1, 1), y_soil.reshape(-1, 1), std_soil.reshape(-1, 1)


if __name__ == '__main__':
    soil_bo(fplot_data=True)