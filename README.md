# Heteroscedastic-BO

Heteroscedastic Bayesian Optimisation uisng the most likely heteroscedatic Gaussian Process as the surrogate model.
Experiment scripts may be found in the GP/toy_experiments directory.

<p align="center">
  <img src="heteroscedastic_gp.gif" width="500" title="logo">
</p>

## Install

We recommend using a conda environment

```
conda create -n hetbo python==3.7
conda activate hetbo
conda install matplotlib numpy pytest scikit-learn panda
conda install scipy==1.1.0
conda install -c conda-forge rdkit
```

## Citing

If you find this code useful please consider citing the following paper https://arxiv.org/abs/1910.07779

```
@article{griffiths2019achieving,
  title={Achieving Robustness to Aleatoric Uncertainty with Heteroscedastic Bayesian Optimisation},
  author={Griffiths, Ryan-Rhys and Garcia-Ortegon, Miguel and Aldrick, Alexander A and Lee, Alpha A},
  journal={arXiv preprint arXiv:1910.07779},
  year={2019}
}
```
