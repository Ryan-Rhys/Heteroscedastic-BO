# Copyright Lee Group 2019
# Author: Ryan-Rhys Griffiths
"""
This script generates the lidar dataset from the file lidar.txt found in the SemiPar package for R:
https://cran.r-project.org/web/packages/SemiPar/index.html
"""

import numpy as np
import pickle


if __name__ == "__main__":
    latitude = []
    longitude = []
    tot_catch = []
    with open('Scallop_Dataset/scallop.txt', 'r') as f:
        i = 1
        for line in f:
            if i != 1:
                data = line.split()
                latitude.append(float(data[0]))
                longitude.append(float(data[1]))
                tot_catch.append(float(data[2]))
            i += 1
    latitude = np.array(latitude).reshape(-1, 1)
    longitude = np.array(longitude).reshape(-1, 1)
    coords = np.concatenate([latitude, longitude], axis=1)
    tot_catch = np.array(tot_catch)
    with open('Scallop_Dataset/coords.pickle', 'wb') as handle:
        pickle.dump(coords, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('Scallop_Dataset/tot_catch.pickle', 'wb') as handle:
        pickle.dump(tot_catch, handle, protocol=pickle.HIGHEST_PROTOCOL)
