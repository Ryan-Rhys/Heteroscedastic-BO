# Copyright Lee Group 2019
# Author: Ryan-Rhys Griffiths
"""
This script generates the lidar dataset from the file lidar.txt found in the SemiPar package for R:
https://cran.r-project.org/web/packages/SemiPar/index.html
"""

import numpy as np
import pickle


if __name__ == "__main__":
    range = []
    logratio = []
    with open('Lidar_1994_Dataset/lidar.txt', 'r') as f:
        i = 1
        for line in f:
            if i != 1:
                data = line.split()
                range.append(float(data[0]))
                logratio.append(float(data[1]))
            i += 1
    range = np.array(range)
    logratio = np.array(logratio)
    with open('Lidar_1994_Dataset/range.pickle', 'wb') as handle:
        pickle.dump(range, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('Lidar_1994_Dataset/logratio.pickle', 'wb') as handle:
        pickle.dump(logratio, handle, protocol=pickle.HIGHEST_PROTOCOL)
