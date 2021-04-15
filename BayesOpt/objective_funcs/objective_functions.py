# Copyright Ryan-Rhys Griffiths 2019
# Author: Ryan-Rhys Griffiths
"""
This module contains objective functions for Bayesian Optimisation.
"""

from matplotlib import pyplot as plt
import matplotlib.cm as cm
import numpy as np
from scipy.optimize import minimize


def linear_sin_noise(X, noise, plot_sample, coefficient, fplot=True, one_off=False):
    """
    1D noise function defined where noise increases linearly in the input domain. Bounds for a bimodal function could be
    [0, 3*pi]

    :param X: input dimension
    :param noise: noise level coefficient for linearly increasing noise
    :param plot_sample: Sample for plotting purposes (points in the input domain)
    :param coefficient: Has the effect of making the maximum with larger noise larger
    :param fplot: Boolean indicating whether to plot the objective, samples and noise function
    :return: f(X) + noise(X)
    """

    noise_function = noise * plot_sample
    linear_sin_noise = np.sin(X) + coefficient*X + 3 + (noise * np.random.randn(*X.shape) * X)
    plot_sin_function = np.sin(plot_sample) + coefficient*plot_sample + 3

    if fplot:
        plt.plot(plot_sample, noise_function, color='green', label='Noise Function', linewidth=4)
        plt.xlabel('x', fontsize=16)
        plt.ylabel('g(x)', fontsize=16)
        #plt.title('Noise Function', fontsize=18)
        plt.tick_params(labelsize=14)
        plt.ylim(-3, 9)
        plt.xlim(0, 10)
        plt.savefig('toy_figures/noise_function.png')
        plt.savefig('toy_one_off_figures/noise_function.png')
        plt.close()
        plt.cla()

        plt.plot(plot_sample, plot_sin_function, color='blue', label='latent function', linewidth=4)
        plt.plot(X, linear_sin_noise, '+', color='green', markersize='12', linewidth='8', label='samples with heteroscedastic noise')
        plt.xlabel('x', fontsize=16)
        plt.ylabel('y', fontsize=16)
        plt.title('Noisy Samples', fontsize=18)
        plt.tick_params(labelsize=14)
        plt.ylim(-3, 9)
        plt.xlim(0, 10)
        plt.legend(loc=3)

        if not one_off:
            plt.savefig('toy_figures/samples.png')
        else:
            plt.savefig('toy_one_off_figures/samples.png')
        plt.close()
        plt.cla()

        plt.plot(plot_sample, plot_sin_function, color='blue', label='latent function', linewidth=4)
        plt.xlabel('x', fontsize=16)
        plt.ylabel('f(x)', fontsize=16)
        #plt.title('Latent Function', fontsize=18)
        plt.tick_params(labelsize=14)
        plt.ylim(-3, 9)
        plt.xlim(0, 10)

        if not one_off:
            plt.savefig('toy_figures/latent_func.png')
        else:
            plt.savefig('toy_one_off_figures/latent_func.png')
        plt.close()
        plt.cla()

        if not one_off:
            plt.plot(plot_sample, plot_sin_function - noise*plot_sample, color='purple', label='black box', linewidth=4)
            plt.ylabel('f(x) - g(x)', fontsize=16)
        else:
            plt.plot(plot_sample, plot_sin_function + noise*plot_sample, color='purple', label='black box', linewidth=4)
            plt.ylabel('f(x) + g(x)', fontsize=16)
        plt.xlabel('x', fontsize=16)
        #plt.title('Black-Box Objective', fontsize=18)
        plt.tick_params(labelsize=14)
        plt.ylim(-3, 9)
        plt.xlim(0, 10)
        if not one_off:
            plt.savefig('toy_figures/black_box.png')
        else:
            plt.savefig('toy_one_off_figures/black_box.png')

    return linear_sin_noise


def max_sin_noise_objective(X, noise, coefficient, fplot=True, penalty=1):
    """
    Objective function for maximising objective - aleatoric noise for the sin wave with linear noise. Used for
    monitoring the best value in the optimisation obtained so far.

    :param X: input to evaluate objective; can be an array of values
    :param noise: noise level coefficient
    :param coefficient: Has the effect of making the maximum with larger noise larger
    :param modification: Whether to modify the function to have one maxima lower than the other
    :param fplot: Boolean indicating whether to plot the black-box objective
    :param penalty: weight penalty for noise
    :return: value of the black-box objective that penalises aleatoric noise, value of the noise at X
    """

    noise_value = noise * X  # value of the heteroscedastic noise at the point(s) X
    objective_value = np.sin(X) + coefficient*X + 3
    composite_objective = objective_value - penalty*noise_value + 3

    if fplot:
        plt.plot(X, composite_objective, color='purple', label='objective - aleatoric noise')
        plt.xlabel('x')
        plt.ylabel('objective(x)')
        plt.title('Black-box Objective')
        plt.ylim(-3, 1)
        plt.xlim(0, 10)
        plt.show()

    composite_objective = float(composite_objective)
    noise_value = float(noise_value)

    return composite_objective, noise_value


def max_one_off_sin_noise_objective(X, noise, coefficient, fplot=True, penalty=1):
    """
    Objective function for maximising objective + aleatoric noise (a one-off good value!) for the sin wave with linear
    noise. Used for monitoring the best value in the optimisation obtained so far.

    :param X: input to evaluate objective; can be an array of values
    :param noise: noise level coefficient
    :param coefficient: Has the effect of making the maximum with larger noise larger
    :param fplot: Boolean indicating whether to plot the black-box objective
    :param penalty: penalty for noise.
    :return: value of the black-box objective that penalises aleatoric noise, value of the noise at X
    """

    noise_value = noise * X  # value of the heteroscedastic noise at the point(s) X
    objective_value = np.sin(X) + coefficient*X
    composite_objective = objective_value + penalty*noise_value

    if fplot:
        plt.plot(X, composite_objective, color='purple', label='objective + aleatoric noise')
        plt.xlabel('x')
        plt.ylabel('objective(x)')
        plt.title('Black-box Objective')
        plt.ylim(-3, 1)
        plt.xlim(0, 10)
        plt.show()

    composite_objective = float(composite_objective)
    noise_value = float(noise_value)

    return composite_objective, noise_value


def min_branin_noise_function(x1, x2, standardised=False, penalty=1):
    """
    Objective function for minimising objective + aleatoric noise

    :param x1: first input dimension
    :param x2: second input dimension
    :param standardised: Bool indicating whether to use the standardised Branin function from Picheny et al. 2012:
    https://link.springer.com/article/10.1007/s00158-013-0919-4
    :param penalty: integer indicating the weight penalty for aleatoric noise
    :return: value of the black-box objective that penalises aleatoric noise, aleatoric noise itself
    """

    return branin_plot_function(x1, x2, standardised) + penalty*noise_plot_function(x1, x2, standardised), \
           noise_plot_function(x1, x2, standardised)


def one_off_min_branin_noise_function(x1, x2, standardised=False, penalty=1):
    """
    Objective function for minimising objective - aleatoric noise (one off value)

    :param x1: first input dimension
    :param x2: second input dimension
    :param standardised: Whether to use the standardised Branin function from Picheny et al. 2012
    https://link.springer.com/article/10.1007/s00158-013-0919-4 on [0, 1]^2
    penalty: integer specifying the weight penalty for aleatoric noise.
    :return: value of the black-box objective that penalises aleatoric noise, aleatoric noise itself
    """

    return branin_plot_function(x1, x2, standardised) - penalty*noise_plot_function(x1, x2, standardised), \
           noise_plot_function(x1, x2, standardised)


def branin_function(x1, x2, noise=0.0, standardised=False):
    """
    Definition of the 2D Branin Function.
    :param x1: numpy array of points along the first dimension
    :param x2: numpy array of points along the second dimension
    :param noise: The noise level (defaults to zero)
    :param standardised: Whether to use the standardised Branin function from Picheny et al. 2012:
    https://link.springer.com/article/10.1007/s00158-013-0919-4 on [0, 1]^2
    :return: The Branin(x1, x2) function evaluated at the points specified by x1 and x2 with noise level given by noise
    """

    # Parameters of the function
    a = 1
    b = 5.1 / (4 * np.pi**2)
    c = 5 / np.pi
    r = 6
    s = 10
    t = 1 / (8 * np.pi)

    # reshape inputs
    x1 = x1.reshape(len(x1), 1)
    x2 = x2.reshape(len(x2), 1)

    if standardised:
        x_bar_one = 15*x1 - 5
        x_bar_two = 15*x2
        f = (1/51.95)*((x_bar_two - ((5.1*x_bar_one**2)/(4*np.pi**2)) + (5*x_bar_one/np.pi) - 6)**2 +
                       ((10 - 10/8*np.pi)*np.cos(x_bar_one)) - 44.81)

    else:
        f = a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s  # Compute the Branin(x1, x2) function

    f += noise**2 * np.random.randn(*x1.shape)  # Add noise to the Branin function f.

    return f


def branin_plot_function(x1, x2, standardised=False):
    """
    Function used for plotting contour plot of Branin-Hoo function.
    :param x1: numpy array of points along the first dimension
    :param x2: numpy array of points along the second dimension
    :param standardised: Whether to use the standardised Branin-Function from Picheny et al. 2012:
    https://link.springer.com/article/10.1007/s00158-013-0919-4 on [0, 1]^2
    :return: The Branin(x1, x2) function evaluated at the points specified by x1 and x2
    """

    # Parameters of the function
    a = 1
    b = 5.1 / (4 * np.pi**2)
    c = 5 / np.pi
    r = 6
    s = 10
    t = 1 / (8 * np.pi)

    if standardised:
        x_bar_one = 15*x1 - 5
        x_bar_two = 15*x2
        f = (1/51.95)*((x_bar_two - ((5.1*x_bar_one**2)/(4*np.pi**2)) + (5*x_bar_one/np.pi) - 6)**2 + ((10 - 10/8*np.pi)*np.cos(x_bar_one)) - 44.81)

    else:
        f = a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s  # Compute the Branin(x1, x2) function

    return f


def noise_plot_function(x1, x2, standardised=False):
    """
    Plot of the noise function.

    :param x1: first input dimension
    :param x2: second input dimension
    :param standardised: Whether to use the standardised Branin function from Picheny et al. 2012:
    https://link.springer.com/article/10.1007/s00158-013-0919-4 on [0, 1]^2
    :return: value of the noise_function(x1, x2).
    """
    location = np.array([x1, x2])

    if standardised:
        return 15 - (2.8*x1**2 + 4.8*x2**2)

    else:
        # TODO construct Gaussian basis noise for non-standardised example.
        # min_one = np.array([-np.pi, 12.275])
        # min_two = np.array([np.pi, 2.275])
        # min_three = np.array([9.42478, 2.475])
        # # Catering for case with batch initialisation with grid of points at the begining viz. single point being
        # # collected.
        # if len(np.shape(location)) > 2:
        #     min_one_dist = location.squeeze(-1) - min_one.reshape(-1, 1)
        #     min_two_dist = location.squeeze(-1) - min_two.reshape(-1, 1)
        #     min_three_dist = location.squeeze(-1) - min_three.reshape(-1, 1)
        # else:
        #     min_one_dist = location - min_one.reshape(-1, 1)
        #     min_two_dist = location - min_two.reshape(-1, 1)
        #     min_three_dist = location - min_three.reshape(-1, 1)
        # amp = 500
        # bandwidth = 50
        # basis_func_one = amp*np.exp(-np.linalg.norm(min_one_dist, axis=0)**2/(2*bandwidth))
        # basis_func_two = amp*np.exp(-np.linalg.norm(min_two_dist, axis=0)**2/(2*bandwidth))
        # basis_func_three = amp*np.exp(-np.linalg.norm(min_three_dist, axis=0)**2/(2*bandwidth))

        return 15 - (2.8*x1**2 + 4.8*x2**2)


def heteroscedastic_branin(x1, x2, standardised=False, f_plot=False, penalty=1):
    """
    Definition of a branin function with heteroscedastic noise

    :param x1: numpy array of points along the first dimension
    :param x2: numpy array of points along the second dimension
    :param standardised: Whether to use the standardised Branin function from Picheny et al. 2012:
    https://link.springer.com/article/10.1007/s00158-013-0919-4 on [0, 1]^2
    :return: The Branin(x1, x2) function evaluated at the points specified by x1 and x2 with noise level given by a
             noise function.
    """

    # Parameters of the function

    a = 1
    b = 5.1 / (4 * np.pi**2)
    c = 5 / np.pi
    r = 6
    s = 10
    t = 1 / (8 * np.pi)

    # reshape inputs
    x1 = x1.reshape(len(x1), 1)
    x2 = x2.reshape(len(x2), 1)

    if standardised:
        x_bar_one = 15*x1 - 5
        x_bar_two = 15*x2
        f = (1/51.95)*((x_bar_two - ((5.1*x_bar_one**2)/(4*np.pi**2)) + (5*x_bar_one/np.pi) - 6)**2 + ((10 - 10/8*np.pi)*np.cos(x_bar_one)) - 44.81)

    else:
        f = a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s  # Compute the Branin(x1, x2) function

    f += noise_plot_function(x1, x2, standardised).reshape(-1, 1) * np.random.randn(*x1.shape)  # Add noise to the Branin function f.

    # -f makes the problem into a maximisation problem (consistent with the sin function experiment.

    if f_plot:
        if standardised:
            x1_star = np.arange(0, 1, 0.02)
            x2_star = np.arange(0, 1, 0.02)
        else:
            x1_star = np.arange(-5.0, 10.0, 0.5)
            x2_star = np.arange(0.0, 15.0, 0.5)

        X, Y = np.meshgrid(x1_star, x2_star)
        latent = branin_plot_function(X, Y, standardised=standardised)
        noise_obj = noise_plot_function(X, Y, standardised=standardised)  # using this for plotting at the moment.
        comp_obj = latent + penalty*noise_obj

        plt.cla()
        CS = plt.contourf(X, Y, latent, cmap=cm.inferno)
        CB = plt.colorbar(CS, shrink=0.8, extend='both')
        plt.xlabel('x1', fontsize=14)
        plt.ylabel('x2', fontsize=14)
        plt.tick_params(labelsize=14)
        #plt.title('Latent Function', fontsize=16)
        plt.savefig('toy_branin_figures/branin_latent.png')
        plt.close()

        plt.cla()
        CS = plt.contourf(X, Y, noise_obj, cmap=cm.inferno)
        CB = plt.colorbar(CS, shrink=0.8, extend='both')
        plt.xlabel('x1', fontsize=14)
        plt.ylabel('x2', fontsize=14)
        plt.tick_params(labelsize=14)
        #plt.title('Noise Function', fontsize=16)
        plt.savefig('toy_branin_figures/branin_noise.png')
        plt.close()

        plt.cla()
        CS = plt.contourf(X, Y, comp_obj, cmap=cm.inferno)
        CB = plt.colorbar(CS, shrink=0.8, extend='both')
        plt.xlabel('x1', fontsize=14)
        plt.ylabel('x2', fontsize=14)
        plt.tick_params(labelsize=14)
        #plt.title('Noise Function', fontsize=16)
        plt.savefig('toy_branin_figures/branin_black_box.png')

        plt.close()

    return f


def opt_branin(x):
    """
    function for numerical optimisation.
    """

    x_bar_one = 15 * x[0] - 5
    x_bar_two = 15 * x[1]
    f = (1 / 51.95) * ((x_bar_two - ((5.1 * x_bar_one ** 2) / (4 * np.pi ** 2)) + (5 * x_bar_one / np.pi) - 6) ** 2 + (
                (10 - 10 / 8 * np.pi) * np.cos(x_bar_one)) - 44.81) - 30*(2*x[0] + x[1])  # last term is the noise function

    return f

# Calculate optima of functions using scipy optimiser


if __name__ == '__main__':
    res = minimize(opt_branin, x0=np.array([0.25, 0.25]), method='L-BFGS-B', bounds=((0, 1), (0, 1)))
    print(res.x)
    print(opt_branin(np.array(res.x)))