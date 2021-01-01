# Copyright Ryan-Rhys Griffiths
# Author: Ryan-Rhys Griffiths
"""
Tests for the Most-likely Heteroscedastic GP and BayesOpt modules.
"""

from matplotlib import pyplot as plt
import numpy as np
import pytest
from scipy.optimize import minimize

from sample_gp_prior import compute_confidence_bounds
from kernels import anisotropic_kernel, compute_kernel_matrix_sq_exp, kernel, sq_exp, scipy_kernel, tanimoto_kernel
from mean_functions import zero_mean
from objective_functions import branin_function, heteroscedastic_branin
from gp_utils import multivariate_normal, mvn_sample, neg_log_marg_lik_krasser, \
    posterior_predictive_krasser, nll_fn, neg_log_marg_lik, my_nll_fn, posterior_predictive, nlpd


def test_mvn_sampler():
    """
    Tests mvn_sample against the numpy implementation. Illustrates expected array shapes.
    """
    np.random.seed(2)
    m = np.zeros((10, 1))
    K = np.eye(10)
    y = mvn_sample(m, K)
    y_numpy = np.random.multivariate_normal(m.reshape(len(m),), K)
    y_numpy = y_numpy.reshape(len(y_numpy), 1)
    assert y_numpy.shape == y.shape


def test_sq_exp():
    """
    Test for the squared exponential kernel.
    """
    assert sq_exp(1.0, 1.0, sigma=2.0, lengthscale=5.0) == 4.0


@pytest.mark.parametrize("input1, input2, signal_amp, lengthscale", [
    (3, 4, 2, 5),
    (4, 3, 1, 4),
    (0, 0, 0, 1),
    (10, 10, 10, 4),
    (20, 10, 3, 2),
    (1, 1, 0, 1),
    (4, 5, 1, 0.3)
])
def test_scipy_kernel(input1, input2, signal_amp, lengthscale):
    """
    Test for the scipy kernel.
    """
    x1 = np.array([input1]).reshape(-1, 1)
    x2 = np.array([input2]).reshape(-1, 1)
    x = scipy_kernel(x1, x2, l=lengthscale, sigma_f=signal_amp)
    y = kernel(x1, x2, l=lengthscale, sigma_f=signal_amp)
    y = y[0][0]
    assert np.allclose(x, y)


@pytest.mark.parametrize("input1, input2, signal_amp", [
    (np.array([[1, 1]]), np.array([[2, 1]]), 2),
    (3, 4, 2),
    (4, 3, 1),
    (0, 0, 0),
    (10, 10, 10),
    (20, 10, 3),
    (1, 1, 0),
    (4, 5, 1)
])
def test_tanimoto_kernel(input1, input2, signal_amp):
    """
    Test for the scipy kernel.
    """
    if type(input1) != np.ndarray:
        x1 = np.array([input1]).reshape(-1, 1)
        x2 = np.array([input2]).reshape(-1, 1)
    else:
        x1 = input1
        x2 = input2
    k_scipy = scipy_kernel(x1, x2, l=1, sigma_f=signal_amp)
    k_tanimoto = tanimoto_kernel(x1, x2, sigma_f=signal_amp)

    assert k_tanimoto.shape == k_scipy.shape


@pytest.mark.parametrize("input1, input2, signal_amp, lengthscale", [
    (3, 4, 2, 5),
    (4, 3, 1, 4),
    (0, 0, 0, 1),
    (10, 10, 10, 4),
    (20, 10, 3, 2),
    (1, 1, 0, 1),
    (4, 5, 1, 0.3)
])
def test_sq_exp_kernel_against_krasser(input1, input2, signal_amp, lengthscale):
    """
    Tests that the squared exponential kernel outputs the same values as Martin Krasser's implementation for
    single x-values.
    """
    x1 = np.array([input1]).reshape(-1, 1)
    x2 = np.array([input2]).reshape(-1, 1)
    x = sq_exp(x1, x2, lengthscale=lengthscale, sigma=signal_amp)
    y = kernel(x1, x2, l=lengthscale, sigma_f=signal_amp)
    y = y[0][0]
    assert np.allclose(x, y)


def test_confidence_bounds():
    """
    Tests that the confidence bounds computed make sense for a simple test case.
    """
    mean_vector = np.zeros((10,))
    K = np.ones((10, 10))
    upper, lower = compute_confidence_bounds(mean_vector, K)
    lower = lower*-1
    assert upper.all() == lower.all()


def test_zero_mean():
    """
    Tests that the zero mean function returns zeros.
    """
    x = np.random.randn(5, 5)
    zero_matrix = zero_mean(x)
    assert zero_matrix.all() == 0


@pytest.mark.parametrize("inputs, signal_amp, lengthscale", [
    (np.array([-4, -3, -2, -1, 1]).reshape(-1, 1), 1, 0.3),
    (np.array([1, 2, 3, 4, 5]).reshape(-1, 1), 2, 2)
])
def test_compute_kernel_matrix_sq_exp_against_krasser(inputs, signal_amp, lengthscale):
    """
    Tests that the squared exponential kernel outputs the same values as Martin Krasser's implementation for a
    one-dimensional input vector.
    """
    krasser_kernel = kernel(inputs, inputs, l=lengthscale, sigma_f=signal_amp)
    kernel_matrix = compute_kernel_matrix_sq_exp(inputs, lengthscale, signal_amp)
    assert np.allclose(krasser_kernel, kernel_matrix)


@pytest.mark.parametrize("xs, xs_star, noise, l, sigma_f, mean_func, kernel_func, fplot", [
    (np.arange(-3, 4, 1).reshape(-1, 1), np.arange(-5, 5, 0.2).reshape(50,1), 0.2, 1, 1, zero_mean, anisotropic_kernel, False),
    (np.arange(-3, 4, 1).reshape(-1, 1), np.arange(-5, 5, 0.2).reshape(50,1),  0.2, 0.3, 1, zero_mean, anisotropic_kernel, False),
    (np.arange(-3, 4, 1).reshape(-1, 1), np.arange(-5, 5, 0.2).reshape(50,1),  0.2, 3, 1, zero_mean, anisotropic_kernel, False)
])
def test_posterior_predictive_against_krasser(xs, xs_star, noise, l, sigma_f, mean_func, kernel_func, fplot):
    """
    Tests that the implementation of the BayesOpt posterior predictive distribution is the same as Martin Krasser's
    implementation.
    """
    y = np.sin(xs) + noise * np.random.randn(*xs.shape)

    pred_mean, pred_var, K, L = posterior_predictive(xs, y, xs_star, noise, l, sigma_f, mean_func, kernel_func, full_cov=False)
    pred_mean_krasser, pred_var_krasser = posterior_predictive_krasser(xs_star, xs, y, l, sigma_f, sigma_y=noise)

    if fplot:

        plt.plot(xs_star, pred_mean, '-', color='gray')
        plt.plot(xs, y, '+', color='red')
        upper = pred_mean + 2*np.sqrt(pred_var)
        lower = pred_mean - 2*np.sqrt(pred_var)
        xs_star = xs_star.reshape(len(xs_star))
        plt.fill_between(xs_star, upper, lower, color='gray', alpha=0.2)
        plt.plot(xs_star, pred_mean_krasser, '-', color='gray')
        upper_krasser = pred_mean_krasser.T + 2*np.sqrt(np.diag(pred_var_krasser))
        lower_krasser = pred_mean_krasser.T - 2*np.sqrt(np.diag(pred_var_krasser))
        upper_krasser = upper_krasser.reshape(50,)
        lower_krasser = lower_krasser.reshape(50,)
        plt.fill_between(xs_star, upper_krasser, lower_krasser, color='red', alpha=0.2)
        plt.show()

    assert np.allclose(pred_mean.reshape(50, 1), pred_mean_krasser)
    assert np.allclose(pred_var, np.diag(pred_var_krasser).reshape(pred_mean_krasser.shape))


@pytest.mark.parametrize("noise, mean_func, kernel_func, l, sigma_f", [
    (0.2, zero_mean, anisotropic_kernel, 1.5, 1.5),
    (0.2, zero_mean, anisotropic_kernel, 1, 1),
    (0.2, zero_mean, anisotropic_kernel, 0.3, 1)
])
def test_neg_log_marg_lik_against_krasser(noise, mean_func, kernel_func, l, sigma_f):
    """
    Tests that the computation of the negative log marginal likelihood is correct by checking it against
    Martin Krasser's implementation.
    """
    xs = np.arange(-3, 4, 1).reshape(-1, 1)
    y = np.sin(xs) + noise**2 * np.random.randn(*xs.shape)
    m = len(xs)
    K = kernel_func(xs, xs, l, sigma_f)  # covariance matrix applied to the x-values of the data points
    L = np.linalg.cholesky(K + noise**2 *np.eye(m))  # We compute the Cholesky factor of the covariance matrix with output noise

    neg_log_marg_likelihood = neg_log_marg_lik(xs, y, noise, l, sigma_f, kernel_func, mean_func)
    neg_log_marg_likelihood_krasser = neg_log_marg_lik_krasser(xs, y, noise, l, sigma_f)
    neg_log_marg_likelihood_gpflow = - multivariate_normal(y, mean_func(xs), L)

    assert np.allclose(neg_log_marg_likelihood, neg_log_marg_likelihood_krasser)
    assert np.allclose(neg_log_marg_likelihood_krasser, neg_log_marg_likelihood_gpflow)
    assert np.allclose(neg_log_marg_likelihood, neg_log_marg_likelihood_gpflow)


@pytest.mark.parametrize("noise, kernel_function, mean_function", [
    (0.2, anisotropic_kernel, zero_mean)
])
def test_neg_log_marg_like_optimisation_against_krasser(noise, kernel_function, mean_function):
    """
    Tests that the optimisation of the negative log marginal likelihood is correct by checking it against Martin
    Krasser's implementation.
    """

    xs = np.arange(-3, 4, 1).reshape(-1, 1)
    y = np.sin(xs) + noise**2 * np.random.randn(*xs.shape)

    res = minimize(my_nll_fn(xs, y, noise, kernel_function, mean_function),
                   [1.5, 1.5], bounds=((1e-5, None), (1e-5, None)), method='L-BFGS-B')
    res_krasser = minimize(nll_fn(xs, y, noise), [1.5, 1.5], bounds=((1e-5, None), (1e-5, None)), method='L-BFGS-B')


    l_opt, sigma_f_opt = res.x
    l_opt_krasser, sigma_f_opt_krasser = res_krasser.x

    assert np.allclose(l_opt, l_opt_krasser)
    assert np.allclose(sigma_f_opt, sigma_f_opt_krasser)


@pytest.mark.parametrize("single_lengthscale, lengthscale_list, signal_amp", [
    (2, [2, 2, 2, 2], 1),
    (1, [1, 1, 1, 1], 1),
    (0.2, [0.2, 0.2, 0.2, 0.2], 1),
    (0.5, [0.5, 0.5, 0.5, 0.5], 2),
])
def test_anisotropic_kernel(single_lengthscale, lengthscale_list, signal_amp):
    """
    Tests that the implementation of the anisotropic kernel is correct.
    """
    input = np.random.rand(4, 4)
    krasser_cov = kernel(input, input, single_lengthscale, signal_amp)
    anisotropic_cov = anisotropic_kernel(input, input, lengthscale_list, signal_amp)

    assert np.allclose(krasser_cov, anisotropic_cov)


@pytest.mark.parametrize("input, lengthscale_list, signal_amp", [
    (np.random.rand(4, 4), [1, 2, 3, 4], 1),
    (np.random.rand(1, 1), [1], 1),
    (np.random.rand(3, 3), [0.2, 0.3, 2], 1),
    (np.random.rand(1, 1), [0.5], 2)
])
def test_scipy_kernel_against_anisotropic(input, lengthscale_list, signal_amp):
    """
    Tests the scipy kernel against the anisotopic kernel for multidimensional inputs.
    """
    anisotropic_cov = anisotropic_kernel(input, input, lengthscale_list, signal_amp)
    scipy_cov = scipy_kernel(input, input, lengthscale_list, signal_amp)
    assert np.allclose(anisotropic_cov, scipy_cov)


@pytest.mark.parametrize("input, lengthscale_list, signal_amp", [
    (np.random.rand(4, 4), [1, 2, 3, 4], 1.5),
    (np.random.rand(1, 1), [0.5], 2),
    (np.random.rand(3, 3), [0.2, 2, 0.7], 1),
    (np.random.rand(1, 1), 0.5, 1)
])
def test_anisotropic_kernel_with_different_lengthscales(input, lengthscale_list, signal_amp):
    """
    Tests that the implementation of the anisotropic kernel works for different values of the lengthscale
    per dimension and different input sizes.
    """
    anisotropic_cov = anisotropic_kernel(input, input, lengthscale_list, signal_amp)
    assert np.allclose(np.sum(np.diag(anisotropic_cov)), signal_amp**2 * input.shape[1])


@pytest.mark.parametrize("noise, l, signal_amp, mean_func, kernel_func", [
    (0.2, [1, 1], 1, zero_mean, anisotropic_kernel),
    (0.4, [5, 5], 2, zero_mean, anisotropic_kernel)
])
def test_neg_log_marg_lik_with_branin_function(noise, l, signal_amp, mean_func, kernel_func):
    """
    Tests that the computation of the negative log marginal likelihood is correct by checking it against
    Martin Krasser's implementation.
    """

    # reshape inputs
    x1 = np.arange(-5.0, 10.0, 0.1)
    x2 = np.arange(0.0, 15.0, 0.1)

    xs = np.concatenate((x1.reshape(len(x1), 1), x2.reshape(len(x2), 1)), axis=1)  # Pass the inputs into the kernel in the right shape

    f = branin_function(x1, x2, noise)

    m = len(xs)
    K = kernel_func(xs, xs, l, signal_amp)
    L = np.linalg.cholesky(K + noise**2 * np.eye(m))

    neg_log_marg_likelihood = neg_log_marg_lik(xs, f, noise, l, signal_amp, kernel_func, mean_func)
    neg_log_marg_likelihood_gpflow = - multivariate_normal(f, mean_func(xs), L)

    assert np.allclose(neg_log_marg_likelihood, neg_log_marg_likelihood_gpflow)


@pytest.mark.parametrize("noise, mean_function, kernel_function", [
    (0.2, zero_mean, anisotropic_kernel),
    (0.4, zero_mean, anisotropic_kernel)
])
def test_neg_log_marg_like_optimisation_in_2D(noise, mean_function, kernel_function):
    """
    Tests that the optimisation of the negative log marginal likelihood is correct by checking it against Martin
    Krasser's implementation.
    """

    # reshape inputs
    x1 = np.arange(-5.0, 10.0, 0.1)
    x2 = np.arange(0.0, 15.0, 0.1)

    xs = np.concatenate((x1.reshape(len(x1), 1), x2.reshape(len(x2), 1)), axis=1)  # Pass the inputs into the kernel in the right shape

    y = branin_function(x1, x2, noise)

    # The bounds are very important here as the lengthscales or signal amplitude can't be negative

    res = minimize(my_nll_fn(xs, y, noise, kernel_function, mean_function),
                   [1.5, 1.5, 1.5], bounds=((1e-5, None), (1e-5, None), (1e-5, None)), method='L-BFGS-B')

    l_opt1, l_opt2, sigma_f_opt = res.x

    assert l_opt1 > 0 and l_opt2 > 0


@pytest.mark.parametrize("pred_mean_vec, pred_var_vec, targets", [
    ([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
    (np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
     np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]))
])
def test_nlpd(pred_mean_vec, pred_var_vec, targets):
    """
    Tests the that nlpd function returns the correct log predictive density. The test case given above corresponds to a
    standard Gaussian ~ (0, 1) where the predictive means are all zero and the targets are all 1. The value is given by
    1/np.sqrt(2*np.e*np.pi) as demonstrated here:

    https://www.wolframalpha.com/input/?x=0&y=0&i=standard+normal+density+at+x+%3D+1+value
    """

    nlpd_val = nlpd(pred_mean_vec, pred_var_vec, targets)

    assert np.allclose(nlpd_val, 1.4189385)


@pytest.mark.parametrize("pred_mean_vec, pred_var_vec, targets", [
    ([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [4, 4, 4, 4, 4, 4, 4, 4, 4, 4], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
])
def test_nlpd_unequal_std_and_variance(pred_mean_vec, pred_var_vec, targets):
    """
    Tests the that nlpd function returns the correct log predictive density.
    There is also an example with a Gaussian ~ (0, 4) where the value at x = 1 is given by:

    https://www.wolframalpha.com/input/?x=0&y=0&i=normal+(0,+2)+density+at+x+%3D+1

    1/(2*np.exp**(1/8.0)*np.sqrt(2*np.pi)) is the form.
    """

    nlpd_val = nlpd(pred_mean_vec, pred_var_vec, targets)

    assert np.allclose(nlpd_val, 1.7370857)


def test_heteroscedastic_branin():
    """
    Tests that the heteroscedastic Branin-Hoo function looks sensible
    """

    x1 = np.arange(-5.0, 10.0, 1.0)
    x2 = np.arange(0.0, 15.0, 1.0)
    xs = np.array(np.meshgrid(x1, x2)).T.reshape(-1, 2)

    y = branin_function(xs[:, 0], xs[:, 1], noise=0.0)
    y_het = heteroscedastic_branin(xs[:, 0], xs[:, 1])

    fplot_data = False

    if fplot_data:

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter(xs[:, 0], xs[:, 1], y, '+', color='red')
        ax.scatter(xs[:, 0], xs[:, 1], y_het, '-', color='blue')
        plt.show()

    assert y.shape == y_het.shape
