# Heteroscedastic-BO

[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Heteroscedastic Bayesian Optimisation uisng the most likely heteroscedatic Gaussian Process as the surrogate model. Implements the approach from "Achieving Robustness to Aleatoric Uncertainty with Heteroscedastic Bayesian Optimisation" available at [https://iopscience.iop.org/article/10.1088/2632-2153/ac298c/meta]

Experiment scripts may be found in the BayesOpt/bayesopt_experiments directory.

<p align="center">
  <img src="heteroscedastic_gp.gif" width="500" title="logo">
</p>

## Install

We recommend using a conda environment

```
conda create -n hetbo python==3.7
conda activate hetbo
conda install matplotlib numpy pytest scikit-learn pandas
conda install scipy==1.1.0
conda install -c conda-forge rdkit
```

## Usage

To reproduce the experiments in the paper using the default values of the hyperparameters

```
python toy_sin_noise.py
python toy_branin_hoo.py
python toy_soil.py
python freesolv.py
python synthetic_func_experiments.py
python gamma_experiments.py
```

To experiment with different hyperparameter settings

```
python toy_sin_noise.py --penalty 1 --aleatoric_weight 1 
                        --random_trials 50 --bayes_opt_iters 5 
```

To adapt the algorithm to your own dataset follow the `toy_soil.py` template making use of the dataloder as
per your task requirements.

## Citing

If you find this code useful please consider citing the following paper [https://iopscience.iop.org/article/10.1088/2632-2153/ac298c/meta]

```
@article{griffiths2021achieving,
  title={Achieving robustness to aleatoric uncertainty with heteroscedastic Bayesian optimisation},
  author={Griffiths, Ryan-Rhys and Aldrick, Alexander A and Garcia-Ortegon, Miguel and Lalchand, Vidhi and others},
  journal={Machine Learning: Science and Technology},
  volume={3},
  number={1},
  pages={015004},
  year={2021},
  publisher={IOP Publishing}
}
```
