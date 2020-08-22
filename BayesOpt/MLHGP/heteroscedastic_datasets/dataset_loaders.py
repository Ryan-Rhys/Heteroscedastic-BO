# Copyright Ryan-Rhys Griffiths 2019
# Author: Ryan-Rhys Griffiths
"""
This script contains functions that generate the benchmark datasets featured in Kersting et al.
http://people.csail.mit.edu/kersting/papers/kersting07icml_mlHetGP.pdf
"""

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import numpy as np
import pickle


class DatasetLoader():
    """
    Class for loading benchmark datasets from Kersting et al. 2007 with additional soil and quasar datasets.
    """

    def __init__(self, dataset, fplot_data):
        """
        :param dataset: str specifying the name of the dataset.
                        One of ['scallop', 'silverman', 'yuan', 'williams', 'goldberg', 'lidar']
        :param fplot_data: bool indicating whether to plot the dataset
        """

        self.dataset = dataset
        self.plot = fplot_data

        np.random.seed(1)  # for the randomly-generated datasets ensure they are always the same

    def load_data(self):
        """
        Load one of the heteroscedastic datasets specified by self.dataset
        """

        if self.dataset == 'yuan':

            # Construct the Yuan 2004 Dataset.

            x = np.random.uniform(0, 1, 200)
            mean = 2 * (np.exp(-30 * (x - 0.25) ** 2) + np.sin(2 * np.pi * x ** 2)) - 2
            std = np.exp(np.sin(2 * np.pi * x))
            y = []

            for i in range(200):
                target = np.random.normal(mean[i], std[i])
                y.append(target)

            # We permute the dataset so that inputs are ordered from left to right across the x-axis

            permutation = x.argsort()
            x = x[permutation]
            y = np.array(y)[permutation]

            x = x.reshape(len(x), 1)
            y = y.reshape(len(y), 1)

            if self.plot:
                plt.plot(x, y, '+', color='green', markersize='12', linewidth='8')
                plt.xlabel('x')
                plt.ylabel('y')
                plt.title('Yuan 2004 Dataset')
                plt.show()

        elif self.dataset == 'goldberg':

            # Construct the Goldberg 1998 dataset.

            x = np.random.uniform(0, 1, 60)
            mean = 2 * np.sin(2 * np.pi * x)
            std = x + 0.5
            y = []
            for i in range(60):
                target = np.random.normal(mean[i], std[i])
                y.append(target)

            # We permute the dataset so that inputs are ordered from left to right across the x-axis

            permutation = x.argsort()
            x = x[permutation]
            y = np.array(y)[permutation]

            if self.plot:
                plt.plot(x, y, '+', color='green', markersize='12', linewidth='8')
                plt.xlabel('x')
                plt.ylabel('y')
                plt.title('Goldberg 1998 Dataset')
                plt.show()

            x = x.reshape(len(x), 1)
            y = y.reshape(len(y), 1)

        elif self.dataset == 'williams':

            # Construct the Williams 1996 dataset.

            x = np.random.uniform(0, np.pi, 200)  # 200 points drawn from a uniform distribution on [0, pi]
            mean = np.sin(2.5 * x) * np.sin(1.5 * x)
            std = 0.01 + 0.25 * (1 - np.sin(2.5 * x)) ** 2
            y = []

            for i in range(200):
                target = np.random.normal(mean[i], std[i])
                y.append(target)

            # We permute the dataset so that inputs are ordered from left to right across the x-axis

            permutation = x.argsort()
            x = x[permutation]
            y = np.array(y)[permutation]

            if self.plot:
                plt.plot(x, y, '+', color='green', markersize='12', linewidth='8')
                plt.xlabel('x')
                plt.ylabel('y')
                plt.title('Williams 1996 Dataset')
                plt.show()

            x = x.reshape(len(x), 1)
            y = y.reshape(len(y), 1)

        elif self.dataset == 'lidar':

            # Constructs the Lidar 1994 dataset.

            with open('../heteroscedastic_datasets/lidar/x.pickle', 'rb') as handle:
                x = pickle.load(handle)
            with open('../heteroscedastic_datasets/lidar/y.pickle', 'rb') as handle:
                y = pickle.load(handle)

            if self.plot:
                plt.plot(x, y, '+', color='green', markersize='12', linewidth='8')
                plt.xlabel('Range')
                plt.ylabel('Logratio')
                plt.title('Lidar 1994 Dataset')
                plt.show()

            x = x.reshape(len(x), 1)
            y = y.reshape(len(y), 1)

        elif self.dataset == 'silverman':

            # Constructs the Silverman motorcycle dataset (1985).

            with open('../heteroscedastic_datasets/silverman/x.pickle', 'rb') as handle:
                x = pickle.load(handle)
            with open('../heteroscedastic_datasets/silverman/y.pickle', 'rb') as handle:
                y = pickle.load(handle)

            if self.plot:
                plt.plot(x, y, '+', color='green', markersize='12', linewidth='8')
                plt.xlim(0, 60)
                plt.ylim(-200, 100)
                plt.xlabel('Times(ms)')
                plt.ylabel('Acceleration(g)')
                plt.title('Silverman Motorcycle Dataset')
                plt.show()

            x = x.reshape(len(x), 1)  # inputs (x)
            y = y.reshape(len(y), 1)  # labels (y)

        else:

            # Constructs the Scallop catch dataset.

            with open('../heteroscedastic_datasets/scallop/x.pickle', 'rb') as handle:
                x = pickle.load(handle)
            with open('../heteroscedastic_datasets/scallop/y.pickle', 'rb') as handle:
                y = pickle.load(handle)

            y = y.reshape(-1, 1)

            if self.plot:
                latitude, longitude = np.meshgrid(x[:, 0], x[:, 1])
                fig = plt.figure()
                ax = plt.axes(projection='3d')
                ax.scatter(x[:, 0], x[:, 1], y, '+', color='red')
                ax.set_xlabel('latitude')
                ax.set_ylabel('longitude')
                ax.set_zlabel('total catch')
                plt.title('The Scallop Dataset')
                plt.show()

        return x, y
