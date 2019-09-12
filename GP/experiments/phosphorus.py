import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

from acquisition_functions import heteroscedastic_expected_improvement, heteroscedastic_propose_location, \
    my_propose_location, my_expected_improvement
from objective_functions import max_p_inorg_objective
from utils import load_p_inorg, get_y_and_std_from_x, plot_BO_progress, get_x_idx
from bo_gp_fit_predict import bo_fit_homo_gp, bo_predict_homo_gp, bo_fit_hetero_gp, bo_predict_hetero_gp


if __name__ == '__main__':

    X_ref, Y_ref, U_ref = load_p_inorg()

    # Number of initialization locations
    n_init = 1
    n_total = len(X_ref)

    # Number of iterations
    n_total = len(X_ref)
    bayes_opt_iters = n_total-n_init


    # We perform random trials of Bayesian Optimisation

    homo_running_sum = np.zeros(bayes_opt_iters)
    homo_squares = np.zeros(bayes_opt_iters)  # Following the single-pass estimator given on pg. 192 of mathematics for machine learning
    hetero_running_sum = np.zeros(bayes_opt_iters)
    hetero_squares = np.zeros(bayes_opt_iters)

    random_trials = 50

    my_het_obj = np.zeros((random_trials,bayes_opt_iters))
    my_homo_obj = np.zeros((random_trials,bayes_opt_iters))


    for i in range(random_trials):

        numpy_seed = i + 50
        np.random.seed(numpy_seed)

        #  Initial noisy data points sampled uniformly at random from the input space.
        n_ref_dataset = X_ref.shape[0]
        init_idx = np.random.randint(0,n_ref_dataset,n_init)
        X_init = X_ref[init_idx]  # sample 3 points at random from the bounds to initialise with
        plot_sample = np.linspace(X_ref[0],X_ref[-1],100).reshape(-1,1)
        Y_init = Y_ref[init_idx]

        # Initialize samples
        homo_X_sample = X_init.reshape(-1, 1)
        homo_Y_sample = Y_init.reshape(-1, 1)
        het_X_sample = X_init.reshape(-1, 1)
        het_Y_sample = Y_init.reshape(-1, 1)

        # initial GP hypers
        # GP1
        l_init = 1.0
        sigma_f_init = 1.0
        gp1_noise = 1.0

        # GP2
        l_noise_init = 1.0
        sigma_f_noise_init = 1.0
        gp2_noise = 1.0

        num_iters = 10
        sample_size = 100

        homo_best_so_far = -300  # value to beat
        het_best_so_far = -300
        homo_obj_val_list = []
        het_obj_val_list = []
        homo_noise_val_list = []
        het_noise_val_list = []
        homo_collected_x = []
        het_collected_x = []
        
        

        for j in range(bayes_opt_iters):

            print('Random trial',i,'Bayes iter',j)

            # Obtain next sampling point from the acquisition function (expected_improvement)
            idx_not_again = get_x_idx(homo_X_sample,X_ref)
            idx_to_evaluate = np.ones(X_ref.shape[0])
            idx_to_evaluate[idx_not_again] = 0
            X_to_evaluate = X_ref[idx_to_evaluate == 1]
            homo_X_next = my_propose_location(my_expected_improvement, homo_X_sample, homo_Y_sample, gp1_noise, l_init, sigma_f_init,
                                            plot_sample, n_restarts=3, min_val=300,x_to_evaluate=X_to_evaluate)

            homo_collected_x.append(homo_X_next)

            # Obtain next noisy sample from the objective function
            homo_Y_next, _ = get_y_and_std_from_x(homo_X_next,load_p_inorg)
            homo_composite_obj_val = max_p_inorg_objective(homo_X_next)

            if homo_composite_obj_val > homo_best_so_far:
                homo_best_so_far = homo_composite_obj_val
                homo_obj_val_list.append(homo_composite_obj_val)
                my_homo_obj[i,j] = homo_composite_obj_val
                
            else:
                homo_obj_val_list.append(homo_best_so_far)
                my_homo_obj[i,j] = homo_best_so_far

            # Add sample to previous samples
            homo_X_sample = np.vstack((homo_X_sample, homo_X_next))
            homo_Y_sample = np.vstack((homo_Y_sample, homo_Y_next))
            
            
            idx_not_again = get_x_idx(het_X_sample,X_ref)
            idx_to_evaluate = np.ones(X_ref.shape[0])
            idx_to_evaluate[idx_not_again] = 0
            X_to_evaluate = X_ref[idx_to_evaluate == 1]
            het_X_next = heteroscedastic_propose_location(heteroscedastic_expected_improvement, het_X_sample, het_Y_sample, gp1_noise, l_init,
                                                    sigma_f_init, l_noise_init, sigma_f_noise_init, gp2_noise, num_iters,
                                                    sample_size, plot_sample, n_restarts=3, min_val=300,x_to_evaluate=X_to_evaluate)

            het_collected_x.append(het_X_next)

            # Obtain next noisy sample from the objective function
            het_Y_next, _ = get_y_and_std_from_x(het_X_next,load_p_inorg)
            het_composite_obj_val = max_p_inorg_objective(het_X_next)

            if het_composite_obj_val > het_best_so_far:
                het_best_so_far = het_composite_obj_val
                het_obj_val_list.append(het_composite_obj_val)
                my_het_obj[i,j] = het_composite_obj_val
            else:
                het_obj_val_list.append(het_best_so_far)
                my_het_obj[i,j] = het_best_so_far

            # Add sample to previous samples
            het_X_sample = np.vstack((het_X_sample, het_X_next))
            het_Y_sample = np.vstack((het_Y_sample, het_Y_next))
        
        this_homo_means = my_homo_obj[:i+1].mean(0)
        this_het_means = my_het_obj[:i+1].std(0)
        
        this_homo_std = my_homo_obj[:i+1].std(0)
        this_het_std = my_het_obj[:i+1].std(0)
        
        print(my_homo_obj[i])
        print(my_het_obj[i])
        
        plot_BO_progress(my_homo_obj[:i+1].mean(0),my_homo_obj[:i+1].std(0),my_het_obj[:i+1].mean(0),my_het_obj[:i+1].std(0))
        
    plot_BO_progress(my_homo_obj[:i+1].mean(0),my_homo_obj[:i+1].std(0),
                    my_het_obj[:i+1].mean(0),my_het_obj[:i+1].std(0),filepath='phosphorus_BO.pdf')
    