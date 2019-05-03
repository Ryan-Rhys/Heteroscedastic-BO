# Copyright Lee Group 2019
# Author: Ryan-Rhys Griffiths
"""
This script contains functions that generate the benchmark datasets featured in Kersting et al.
http://people.csail.mit.edu/kersting/papers/kersting07icml_mlHetGP.pdf
"""

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import numpy as np
import pickle


def silverman_1985(fplot_data=False):
    """
    Constructs the Silverman motorcycle dataset (1985).

    :param fplot_data: Boolean indicating whether or not to plot the dataset
    :return: inputs (times) and targets (acceleration) of the Silverman dataset
    """

    with open('Silverman_Motorcycle_Dataset/times.pickle', 'rb') as handle:
        times = pickle.load(handle)
    with open('Silverman_Motorcycle_Dataset/accel.pickle', 'rb') as handle:
        accel = pickle.load(handle)

    if fplot_data:
        plt.plot(times, accel, '+', color='green', markersize='12', linewidth='8')
        plt.xlim(0, 60)
        plt.ylim(-200, 100)
        plt.xlabel('Times(ms)')
        plt.ylabel('Acceleration(g)')
        plt.title('Silverman Motorcycle Dataset')
        plt.show()

    times = times.reshape(len(times), 1)  # inputs (x)
    accel = accel.reshape(len(accel), 1)  # labels (y)

    return times, accel


def lidar_1994(fplot_data=False):
    """
    Constructs the Lidar 1994 dataset.

    :param fplot_data: Boolean indicating whether or not to plot the dataset
    :return: inputs and targets of the Lidar dataset.
    """

    with open('Lidar_1994_Dataset/range.pickle', 'rb') as handle:
        range = pickle.load(handle)
    with open('Lidar_1994_Dataset/logratio.pickle', 'rb') as handle:
        logratio = pickle.load(handle)

    if fplot_data:
        plt.plot(range, logratio, '+', color='green', markersize='12', linewidth='8')
        plt.xlabel('Range')
        plt.ylabel('Logratio')
        plt.title('Lidar 1994 Dataset')
        plt.show()

    range = range.reshape(len(range), 1)  # inputs (x)
    logratio = logratio.reshape(len(logratio), 1)  # labels (y)

    return range, logratio


def scallop_data(fplot_data=False):
    """
    Constructs the Scallop catch dataset.

    :param fplot_data: Boolean indicating whether or not to plot the dataset
    :return: inputs and targets of the Scallop Catch dataset.
    """

    with open('/Users/Ryan-Rhys/ml_physics/perovskite-bayesopt/GP/Scallop_Dataset/coords.pickle', 'rb') as handle:
        coords = pickle.load(handle)
    with open('/Users/Ryan-Rhys/ml_physics/perovskite-bayesopt/GP/Scallop_Dataset/tot_catch.pickle', 'rb') as handle:
        tot_catch = pickle.load(handle)

    tot_catch = tot_catch.reshape(-1, 1)

    if fplot_data:

        latitude, longitude = np.meshgrid(coords[:, 0], coords[:, 1])

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter(coords[:, 0], coords[:, 1], tot_catch, '+', color='red')
        ax.set_xlabel('latitude')
        ax.set_ylabel('longitude')
        ax.set_zlabel('total catch')
        plt.title('The Scallop Dataset in its Fully Sparse Glory')
        plt.show()

    return coords, tot_catch


def williams_1996(fplot_data=False):
    """
    Constructs the Williams (1996) dataset.

    :param fplot_data: Boolean indicating whether or not to plot the dataset.
    :return: inputs and targets of Williams dataset
    """

    np.random.seed(1)  # making sure the generated dataset is the same every time.

    inputs = np.random.uniform(0, np.pi, 200)  # 200 points drawn from a uniform distribution on [0, pi]
    mean = np.sin(2.5*inputs)*np.sin(1.5*inputs)
    std = 0.01 + 0.25*(1 - np.sin(2.5*inputs))**2
    targets = []
    for i in range(200):
        target = np.random.normal(mean[i], std[i])
        targets.append(target)

    # We permute the dataset so that inputs are ordered from left to right across the x-axis

    permutation = inputs.argsort()
    inputs = inputs[permutation]
    targets = np.array(targets)[permutation]

    if fplot_data:
        plt.plot(inputs, targets, '+', color='green', markersize='12', linewidth='8')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Williams 1996 Dataset')
        plt.show()

    inputs = inputs.reshape(len(inputs), 1)
    targets = targets.reshape(len(targets), 1)

    return inputs, targets


def goldberg_1998(fplot_data=False):
    """
    Constructs the Goldberg 1998 dataset.

    :param fplot_data: Boolean indicating whether or not to plot the dataset
    :return: inputs and targets of the Goldberg dataset.
    """

    np.random.seed(1)  # making sure the generated dataset is the same each time.

    inputs = np.random.uniform(0, 1, 60)
    mean = 2*np.sin(2*np.pi*inputs)
    std = inputs + 0.5
    targets = []
    for i in range(60):
        target = np.random.normal(mean[i], std[i])
        targets.append(target)

    # We permute the dataset so that inputs are ordered from left to right across the x-axis

    permutation = inputs.argsort()
    inputs = inputs[permutation]
    targets = np.array(targets)[permutation]

    if fplot_data:
        plt.plot(inputs, targets, '+', color='green', markersize='12', linewidth='8')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Goldberg 1998 Dataset')
        plt.show()

    inputs = inputs.reshape(len(inputs), 1)
    targets = targets.reshape(len(targets), 1)

    return inputs, targets


def yuan_2004(fplot_data=False):
    """
    Constructs the Yuan 2004 dataset.

    :param fplot_data: Boolean indicating whether or not to plot the dataset
    :return: inputs and targets of the Yuan dataset.
    """

    np.random.seed(1)

    inputs = np.random.uniform(0, 1, 200)
    mean = 2*(np.exp(-30*(inputs - 0.25)**2) + np.sin(2*np.pi*inputs**2)) - 2
    std = np.exp(np.sin(2*np.pi*inputs))
    targets = []
    for i in range(200):
        target = np.random.normal(mean[i], std[i])
        targets.append(target)

    # We permute the dataset so that inputs are ordered from left to right across the x-axis

    permutation = inputs.argsort()
    inputs = inputs[permutation]
    targets = np.array(targets)[permutation]

    if fplot_data:
        plt.plot(inputs, targets, '+', color='green', markersize='12', linewidth='8')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Yuan 2004 Dataset')
        plt.show()

    inputs = inputs.reshape(len(inputs), 1)
    targets = targets.reshape(len(targets), 1)

    return inputs, targets
