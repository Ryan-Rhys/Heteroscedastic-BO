# Copyright Ryan-Rhys Griffiths 2019
# Author: Ryan-Rhys Griffiths
"""
This script generates the heteroscedastic datasets found in Kersting et al. 2007
(http://people.csail.mit.edu/kersting/papers/kersting07icml_mlHetGP.pdf)

Lidar dataset from the file lidar.txt found in the SemiPar package for R:
https://cran.r-project.org/web/packages/SemiPar/index.html

Silverman dataset from the file silverman.txt copied from:
http://www.stat.cmu.edu/~larry/all-of-statistics/=data/motor.dat
"""

import argparse
import pickle

import numpy as np


def main(dataset):
    """
    Generate the heteroscedastic dataset given by the 'dataset' identifier.

    :param dataset: str specifying the dataset to generate. One of ['silverman', 'scallop', 'lidar'].
    """

    if dataset == 'scallop':

        # 2-D dataset

        # Store the (x1, x2, y) coordinates of the heteroscedastic data. x1 = latitude, x2 = longitude, y = total catch

        x1 = []
        x2 = []
        y = []

        with open(f'{dataset}/{dataset}.txt', 'r') as f:
            i = 1
            for line in f:
                if i != 1:
                    data = line.split()
                    x1.append(float(data[0]))
                    x2.append(float(data[1]))
                    y.append(float(data[2]))
                i += 1

        x1 = np.array(x1).reshape(-1, 1)
        x2 = np.array(x2).reshape(-1, 1)
        x = np.concatenate([x1, x2], axis=1)
        y = np.array(y)

        with open(f'{dataset}/x.pickle', 'wb') as handle:
            pickle.dump(x, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f'{dataset}/y.pickle', 'wb') as handle:
            pickle.dump(y, handle, protocol=pickle.HIGHEST_PROTOCOL)

    else:

        # 1-D dataset.

        # Store the (x, y) coordinates of the heteroscedastic data

        x = []
        y = []

        with open(f'{dataset}/{dataset}.txt', 'r') as f:
            i = 1
            for line in f:
                if i != 1:
                    data = line.split()
                    x.append(float(data[0]))
                    y.append(float(data[1]))
                i += 1

        x = np.array(x)
        y = np.array(y)

        with open(f'{dataset}/x.pickle', 'wb') as handle:
            pickle.dump(x, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f'{dataset}/y.pickle', 'wb') as handle:
            pickle.dump(y, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--dataset', type=str, default='silverman',
                        help='Heteroscedastic dataset to generate. Choices are one of '
                             '[lidar, silverman, scallop]')

    args = parser.parse_args()

    main(args.dataset)
