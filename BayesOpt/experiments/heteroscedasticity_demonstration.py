# Copyright Ryan-Rhys Griffiths 2019
# Author: Ryan-Rhys Griffiths
"""
This script demonstrates a proof of concept for the presence of heteroscedasticity in neural architecture search with
respect to different validation sets. The experiments may be reproduced using the parameters supplied here.
"""

from sklearn.datasets import load_boston, load_breast_cancer
from sklearn import preprocessing

from exp_utils import measure_reg_performance, measure_class_performance


i_reg = False  # Indicates whether the task is classification or regression


if __name__ == '__main__':

    if i_reg:
        dataset = load_boston()
    else:
        dataset = load_breast_cancer()

    data, target = dataset.data, dataset.target
    data_scaler = preprocessing.MinMaxScaler()
    data = data_scaler.fit_transform(data)

    # For the regression task we scale the target values as well.

    if i_reg:
        target_scaler = preprocessing.MinMaxScaler()
        target = target_scaler.fit_transform(target)

    target = target.reshape(-1, 1)
    n_inputs = data.shape[1]
    n_outputs = target.shape[1]

    # We specify the number of folds of k-fold cross-validation and the number of random trials (random k-fold splits)

    k_folds = 3
    n_trials = 20

    # We specify the network parameters whose noise we want to assess

    if i_reg:
        num_neurons = 50
        mu = 0.2
        av_trial_performance, av_trial_noise = measure_reg_performance(data, target, num_neurons, mu, k_folds, n_trials, target_scaler)
    else:
        num_neurons = 2
        num_layers = 3
        av_trial_performance, av_trial_noise = measure_class_performance(data, target, num_neurons, num_layers, k_folds, n_trials)  # optimiser needs to be changed to conjugate gradients in measure_class_performance to reproduce the results from the paper.

    print("average trial performance is:" + str(av_trial_performance))
    print("average trial noise is:" + str(av_trial_noise))
