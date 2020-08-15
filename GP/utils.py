# Copyright Lee Group 2019
# Author: Ryan-Rhys Griffiths
"""
Functions from other open source GP libraries used for test purposes.
"""

from matplotlib import pyplot as plt
import numpy as np
import scipy.stats
from scipy.linalg import cholesky, inv, solve_triangular

from kernels import kernel, anisotropic_kernel, scipy_kernel
from mean_functions import zero_mean


def posterior_predictive(xs, y, xs_star, noise, l, sigma_f, mean_func=zero_mean, kernel=anisotropic_kernel, full_cov=True):
    """
    Compute the posterior predictive mean and variance of the GP.

    :param xs: training data input locations
    :param y: training data targets
    :param xs_star: test data input locations
    :param noise: noise level
    :param l: kernel lengthscale
    :param sigma_f: signal amplitude
    :param mean_func: prior mean function
    :param kernel: GP covariance function
    :return: pred_mean, pred_var, K, L; the GP posterior predictive mean and variance, training covariance matrix and
             the Cholesky decomposition of the covariance matrix.
    """

    jitter = 1e-3

    m = len(xs)  # number of training points
    mean_vector = mean_func(xs)  # mean function applied to the training inputs
    K = kernel(xs, xs, l, sigma_f)  # covariance matrix applied to the x-values of the data points
    L = np.linalg.cholesky(K + noise**2 * np.eye(m))  # We compute the Cholesky factor of the covariance matrix with output noise
    K_ss = kernel(xs_star, xs_star, l, sigma_f)  # Using Katherine Bailey's notation for the cov matrix at test locations
    K_s = kernel(xs, xs_star, l, sigma_f)
    Lk = np.linalg.solve(L, K_s)
    pred_mean = np.dot(Lk.T, np.linalg.solve(L, y - mean_vector))
    pred_var = K_ss - np.dot(Lk.T, Lk)

    assert np.diag(pred_var).all() >= 0

    if not full_cov:
        pred_var = np.diag(pred_var)
        pred_var = pred_var.reshape(pred_mean.shape)

    pred_var += (jitter * np.eye(pred_var.shape[0]))

    return pred_mean, pred_var, K, L


def mvn_sample(mean_vector, K, jitter=1e-8):
    """
    Sample from a multivariate normal distribution. Rasmussen and Williams page 201.

    :param mean_vector: numpy array giving the mean vector of the multivariate normal
    :param K: numpy array giving the covariance matrix of the multivariate normal
    :param jitter: float giving the amount of jitter to add to the covariance matrix
    :return: a single sample from the multivariate normal
    """

    dim = len(mean_vector)  # dimensionality of the multivariate normal
    assert dim == len(K[0, :])  # check that dimensions match

    jitter_matrix = np.eye(dim, dim)*jitter
    K += jitter_matrix

    L = np.linalg.cholesky(K)  # Be careful about jitter here
    mu = np.zeros((dim, 1))
    for i in range(0, dim):
        mu[i] = np.random.randn()
    y = mean_vector + L@mu
    return y


def posterior_predictive_krasser(X_s, X_train, Y_train, l, sigma_f, sigma_y=1e-8):
    """
    :param X_s: test input locations
    :param X_train: m training data points
    :param Y_train: (m x 1) training targets
    :param l: kernel lengthscale parameter
    :param sigma_f: signal amplitude parameter
    :param sigma_y: noise parameter
    :return: posterior mean vector (n x d) and covariance matrix (n x n)

    Martin Krasser's implementation of the posterior predictive distribution. Assumes zero mean I think.

    Computes the sufficient statistics of the GP posterior predictive distribution from m training data X_train
    and Y_train and n new inputs X_s. Args: X_s: New input locations (n x d). X_train: Training locations (m x d).
    Y_train: Training targets (m x 1). l: Kernel length parameter. sigma_f: Kernel vertical variation parameter.
    sigma_y: Noise parameter. Returns: Posterior mean vector (n x d) and covariance matrix (n x n).
    """

    K = kernel(X_train, X_train, l, sigma_f) + sigma_y ** 2 * np.eye(len(X_train))
    K_s = kernel(X_train, X_s, l, sigma_f)
    K_ss = kernel(X_s, X_s, l, sigma_f) + 1e-8 * np.eye(len(X_s))
    K_inv = inv(K)

    mu_s = K_s.T.dot(K_inv).dot(Y_train)

    cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)

    assert np.diag(cov_s).all() >= 0

    return mu_s, cov_s


def multivariate_normal(x, mu, L):
    """
    This function is taken from GPFlow and adapted to work with numpy instead of TensorFlow.
    It corresponds to the calculation of the log marginal likelihood in equation 2.30 on pg. 19 of Rasmussen and
    Williams.

    Computes the log-density of a multivariate normal.
    :param x  : Dx1 or DxN sample(s) for which we want the density
    :param mu : Dx1 or DxN mean(s) of the normal distribution
    :param L  : DxD Cholesky decomposition of the covariance matrix
    :return p : (1,) or (N,) vector of log densities for each of the N x's and/or mu's
    x and mu are either vectors or matrices. If both are vectors (N,1):
    p[0] = log pdf(x) where x ~ N(mu, LL^T)
    If at least one is a matrix, we assume independence over the *columns*:
    the number of rows must match the size of L. Broadcasting behaviour:
    p[n] = log pdf of:
    x[n] ~ N(mu, LL^T) or x ~ N(mu[n], LL^T) or x[n] ~ N(mu[n], LL^T)
    """
    # if x.shape.ndims is None:
    #     logger.warning('Shape of x must be 2D at computation.')
    # elif x.shape.ndims != 2:
    #     raise ValueError('Shape of x must be 2D.')
    # if mu.shape.ndims is None:
    #     logger.warning('Shape of mu may be unknown or not 2D.')
    # elif mu.shape.ndims != 2:
    #     raise ValueError('Shape of mu must be 2D.')

    d = x - mu
    alpha = solve_triangular(L, d, lower=True)
    num_dims = np.float64(np.shape(d)[0])
    p = - 0.5 * np.sum(np.square(alpha), 0)
    p -= 0.5 * num_dims * np.log(2 * np.pi)
    p -= np.sum(np.log(np.diag(L)))
    return p


def neg_log_marg_lik(xs, y, noise, l, sigma_f, kernel=anisotropic_kernel, mean_func=zero_mean):
    """
    Compute the negative log marginal likelihood.

    :param xs: training input locations
    :param y: training targets
    :param noise: noise parameter
    :param l: kernel lengthscale list
    :param sigma_f: signal amplitude
    :param kernel: covariance function to use
    :param mean_func: mean function to use
    :return: Value of negative log marginal likelihood
    """
    m = len(xs)
    K = kernel(xs, xs, l, sigma_f)
    mean_vector = mean_func(xs)

    log_marg_lik = -m/2*np.log(2*np.pi) - 1/2*np.log(np.linalg.det(K + noise**2 * np.eye(m))) - \
                    1/2*(y - mean_vector).T@np.linalg.inv(K + noise**2 * np.eye(m))@(y - mean_vector)

    neg_log_marg_lik = - log_marg_lik

    return neg_log_marg_lik


def neg_log_marg_lik_krasser(X_train, Y_train, noise, l, sigma_f):
    """
    Martin Krasser's implementation of the marginal likelihood using scipy kernel instead of the orginal kernel function
    in order to extend functionality to higher dimensions.

    :param X_train: training input locations
    :param Y_train: training targets
    :param noise: noise parameter
    :param l: lengthscale
    :param sigma_f: signal amplitude
    :return: the value of the marginal likelihood
    """
    K = scipy_kernel(X_train, X_train, l, sigma_f) + noise ** 2 * np.eye(len(X_train))

    # Compute determinant via Cholesky decomposition
    neg_log_marg_lik = np.sum(np.log(np.diagonal(cholesky(K)))) + 0.5 * Y_train.T.dot(inv(K).dot(Y_train)) + \
                       0.5 * len(X_train) * np.log(2 * np.pi)
    return neg_log_marg_lik


def my_nll_fn(xs, y, noise, kernel_function, mean_function):
    """
    :param X_train: training inputs locations
    :param Y_train: training targets
    :param noise: noise parameter
    :param kernel_function: covariance function
    :param mean_function: mean function
    :return: optimisation step

    My negative log marginal likelihood computation to be fed into the scipy optimiser.

    Returns a function that computes the negative log-likelihood for training data xs and y and given
    noise level. Args: xs: training locations (m x d). y: training targets (m x 1). noise: known noise
    level of y. Returns: Minimization objective.
    """
    def step(theta):
        return neg_log_marg_lik(xs, y, noise, theta[0:len(theta) - 1], theta[-1], kernel_function, mean_function)
    return step


def nll_fn(X_train, Y_train):
    """
    :param X_train: training inputs locations
    :param Y_train: training targets
    :return: optimisation step

    Martin Krasser's negative log marginal likelihood computation to be fed into the scipy optimiser.

    Returns a function that computes the negative log-likelihood for training data X_train and Y_train and given
    noise level. Args: X_train: training locations (m x d). Y_train: training targets (m x 1).
    Returns: Minimization objective. July 12th - adding noise as an extra optimisation parameter
    """

    jitter = 1e-3  # additive jitter term to prevent numerical instability

    def step(theta):
        K = scipy_kernel(X_train, X_train, l=theta[0:len(theta) - 2], sigma_f=theta[-2]) + theta[-1]**2 + jitter * np.eye(len(X_train))
        # Compute determinant via Cholesky decomposition
        return np.sum(np.log(np.diagonal(cholesky(K)))) + 0.5 * Y_train.T.dot(inv(K).dot(Y_train)) + \
               0.5 * len(X_train) * np.log(2*np.pi)
    return step


def nll_fn_het(X_train, Y_train, noise):
    """
    :param X_train: training inputs locations
    :param Y_train: training targets
    :param noise: fixed noise parameter of y_train
    :return: optimisation step

    Martin Krasser's negative log marginal likelihood computation to be fed into the scipy optimiser.

    Returns a function that computes the negative log-likelihood for training data X_train and Y_train and given
    noise level. Args: X_train: training locations (m x d). Y_train: training targets (m x 1).
    Returns: Minimization objective. For the heteroscedastic GP we don't optimise the noise level. For some reason it
    works better this way.
    """

    def step(theta):
        K = scipy_kernel(X_train, X_train, l=theta[0:len(theta) - 1], sigma_f=theta[-1]) + noise**2 * np.eye(len(X_train))
        # Compute determinant via Cholesky decomposition
        return np.sum(np.log(np.diagonal(cholesky(K)))) + 0.5 * Y_train.T.dot(inv(K).dot(Y_train)) + \
               0.5 * len(X_train) * np.log(2*np.pi)
    return step


def nlpd(pred_mean_vec, pred_var_vec, targets):
    """
    Computes the negative log predictive density for a set of targets assuming a Gaussian noise model.

    :param pred_mean_vec: predictive mean of the model at the target input locations
    :param pred_var_vec: predictive variance of the model at the target input locations
    :param targets: target values
    :return: nlpd (negative log predictive density)
    """
    assert len(pred_mean_vec) == len(pred_var_vec)  # pred_mean_vec must have been evaluated at xs corresponding to ys.
    assert len(pred_mean_vec) == len(targets)

    nlpd = 0
    index = 0
    n = len(targets)  # number of data points

    pred_mean_vec = np.array(pred_mean_vec).reshape(n, )
    pred_var_vec = np.array(pred_var_vec).reshape(n, )
    pred_std_vec = np.sqrt(pred_var_vec)
    targets = np.array(targets).reshape(n, )

    for target in targets:
        density = scipy.stats.norm(pred_mean_vec[index], pred_std_vec[index]).pdf(target)
        nlpd += -np.log(density)
        index += 1

    nlpd /= n

    return nlpd

def plot_het_gp1(xs, ys, xs_star, gp1_noise, gp1_l, gp1_sigma_f):
    """
    Plot GP1 from the heteroscedastic GP.

    :param xs: input locations (m x d)
    :param ys: y values (m x 1)
    :param xs_star: test locations (n x d)
    :param gp1_noise: aleatoric noise
    :param gp1_l: kernel lengthscale
    :param gp1_sigma_f: kernel signal amplitude
    :return: None
    """

    gp1_pred_mean, gp1_pred_var, _, _ = posterior_predictive(xs, ys, xs_star, gp1_noise, gp1_l, gp1_sigma_f,
                                                             mean_func=zero_mean, kernel=scipy_kernel)

    gp1_plot_pred_var = np.diag(gp1_pred_var).reshape(-1, 1)  # Take the diagonal of the covariance matrix for plotting purposes
    # TODO: ADD ALEATORIC NOISE
    gp1_plot_pred_var = gp1_plot_pred_var # + np.square(gp1_noise) - commented out because it causes computational error. need a workaround
    print(np.square(gp1_noise))
    plt.plot(xs, ys, '+', color='green', markersize='12', linewidth='8')
    plt.plot(xs_star, gp1_pred_mean, '-', color='red')
    upper = gp1_pred_mean + 2 * np.sqrt(gp1_plot_pred_var)
    lower = gp1_pred_mean - 2 * np.sqrt(gp1_plot_pred_var)
    upper = upper.reshape(xs_star.shape)
    lower = lower.reshape(xs_star.shape)
    plt.fill_between(xs_star.reshape(len(xs_star), ), upper.reshape(len(xs_star), ), lower.reshape(len(xs_star), ),
                     color='gray', alpha=0.2)
    plt.xlabel('input, x')
    plt.ylabel('f(x)')
    plt.title('Heteroscedastic GP1 Posterior')
    plt.show()

    return None


def plot_het_gp2(xs, variance_estimator, xs_star, gp2_noise, gp2_l, gp2_sigma_f):
    """
    Plot GP2 from the heteroscedastic GP.

    :param xs: input locations (m x d)
    :param variances: sampled noise (m x 1)
    :param xs_star: test locations (n x d)
    :param gp2_noise: fixed noise level
    :param gp2_l: kernel lengthscale
    :param gp2_sigma_f: kernel signal amplitude
    :return: None
    """

    gp2_pred_mean, gp2_pred_var, _, _ = posterior_predictive(xs, variance_estimator, xs_star, gp2_noise, gp2_l, gp2_sigma_f,
                                                             mean_func=zero_mean, kernel=scipy_kernel)

    gp2_plot_pred_var = np.diag(gp2_pred_var).reshape(-1, 1)  # Take the diagonal of the covariance matrix for plotting purposes
    plt.plot(xs, variance_estimator, '+', color='green', markersize='12', linewidth='8')
    plt.plot(xs_star, gp2_pred_mean, '-', color='red')
    upper = gp2_pred_mean + 2 * np.sqrt(gp2_plot_pred_var)
    lower = gp2_pred_mean - 2 * np.sqrt(gp2_plot_pred_var)
    upper = upper.reshape(xs_star.shape)
    lower = lower.reshape(xs_star.shape)
    plt.fill_between(xs_star.reshape(len(xs_star), ), upper.reshape(len(xs_star), ), lower.reshape(len(xs_star), ),
                     color='gray', alpha=0.2)
    plt.xlabel('input, x')
    plt.ylabel('variance(x)')
    plt.title('Heteroscedastic GP2 Posterior')
    plt.show()

    return None


def one_d_train_test_split(xs, ys, split_ratio):
    """
    Splits a dataset of (xs, ys) into train and test sets of a given split ratio.

    :param xs: dataset inputs
    :param ys: dataset outputs
    :param split_ratio: The ratio in which to split train/test. 0 < split_ratio < 1. i.e. 0.9 will be 90/10 train/test.
    :return: xs_train, ys_train, xs_test, ys_test
    """

    n = xs.shape[0]
    permutation = np.random.choice(n, n, replace=False)

    # train/test sets slightly unbalanced in terms of labels due to random partitioning

    xs_train = xs[permutation, :][0: np.int(np.round(split_ratio * n)), :]  # 90/10 train/test split
    xs_test = xs[permutation, :][np.int(np.round(split_ratio * n)):, :]
    ys_train = ys[permutation, :][0: np.int(np.round(split_ratio * n)), :]
    ys_test = ys[permutation, :][np.int(np.round(split_ratio * n)):, :]

    # We permute the dataset so that inputs are ordered from left to right across the x-axis

    xs_train = xs_train.reshape(len(xs_train), )
    xs_test = xs_test.reshape(len(xs_test), )
    ys_train = ys_train.reshape(len(ys_train), )
    ys_test = ys_test.reshape(len(ys_test), )

    permutation_train = xs_train.argsort()
    xs_train = xs_train[permutation_train]
    ys_train = np.array(ys_train)[permutation_train]
    permutation_test = xs_test.argsort()
    xs_test = xs_test[permutation_test]
    ys_test = np.array(ys_test)[permutation_test]

    xs_train = xs_train.reshape(len(xs_train), 1)
    xs_test = xs_test.reshape(len(xs_test), 1)
    ys_train = ys_train.reshape(len(ys_train), 1)
    ys_test = ys_test.reshape(len(ys_test), 1)

    return xs_train, ys_train, xs_test, ys_test
