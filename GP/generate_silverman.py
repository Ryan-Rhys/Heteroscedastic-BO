# Copyright Lee Group 2019
# Author: Ryan-Rhys Griffiths
"""
This script generates the silverman dataset from the file silverman.txt copied from:
http://www.stat.cmu.edu/~larry/all-of-statistics/=data/motor.dat
"""

import numpy as np
import pickle


if __name__ == "__main__":
    times = []
    accel = []
    with open('Silverman_Motorcycle_Dataset/silverman.txt', 'r') as f:
        i = 1
        for line in f:
            if i != 1:
                data = line.split()
                times.append(float(data[0]))
                accel.append(float(data[1]))
            i += 1
    times = np.array(times)
    accel = np.array(accel)
    with open('Silverman_Motorcycle_Dataset/times.pickle', 'wb') as handle:
        pickle.dump(times, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('Silverman_Motorcycle_Dataset/accel.pickle', 'wb') as handle:
        pickle.dump(accel, handle, protocol=pickle.HIGHEST_PROTOCOL)
