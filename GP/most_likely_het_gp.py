import torch
import gpytorch
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import GaussianLikelihood, FixedNoiseGaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.constraints import Positive

def train_a_GP(model, train_x, train_y, likelihood, training_iter):
    """
    Simple utility function to train a Gaussian process (GP) model with Adam (following the examples on the docs)
    :param model: GP model
    :param train_x: tensor with training features X
    :param train_y: tensor with training targets Y
    :param likelihood: likelihood function
    :param training_iter: number of iterations to train
    :return: trained GP model, trained likelihood
    """
    # train GP_model for training_iter iterations
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},  # Includes GaussianLikelihood parameters
    ], lr=0.1)

    # "Loss" for GPs - the marginal log likelihood
    mll = ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, train_y)
        loss.backward()
        print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f' % (
            i + 1, training_iter, loss.item(),
            model.covar_module.base_kernel.lengthscale.item(),
        ))
        optimizer.step()

        model.eval()
        likelihood.eval()
    return model, likelihood


class ExactGPModel(gpytorch.models.ExactGP):
    """
    Exact Gaussian process model (following the examples in the docs).
    """
    def __init__(self, train_x, train_y, likelihood):
        """
        Initializer function. Specifies the mean and the covariance functions.
        :param train_x: tensor with training features X
        :param train_y: tensor with training targets Y
        :param likelihood: likelihood function
        :return: None
        """
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(RBFKernel())

    def forward(self, x):
        """
        Forward method to evaluate GP.
        :param x: tensor with features X on which to evaluate the GP.
        :return: MultivariateNormal
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class hetGPModel():
    """
    Most likely heteroscedastic GP model.
    """
    def __init__(self,train_x,train_y,training_iter=100,het_fitting_iter=10,var_estimator_n=50):
        """
        Initializer function.
        :param train_x: tensor with training features X
        :param train_y: tensor with training targets Y
        :param training_iter: number of iterations to train GP1, GP2 and GP3
        :param het_fitting_iter: number of iterations to run the pseudo expectation maximization (EM) algorithm while refining GP3
        :param var_estimator_n: number of samples to estimate the variance at each training point
        :return: None
        """
        self.train_x = train_x
        self.train_y = train_y
        self.training_iter = training_iter
        self.var_estimator_n = var_estimator_n
        self.het_fitting_iter = het_fitting_iter
        self.final_GP = None
        self.final_lik = None
        self.final_r = None

    def predict(self,x):
        """
        Predict method to evaluate GP.
        :param x: tensor with features X on which to evaluate the GP.
        :return: MultivariateNormal
        """
        if self.final_GP is None:
            raise RuntimeError('hetGPModel needs to be trained before using it')
        return self.final_GP(x)

    def train_model(self):
        # train self.GP1 if self.is_GP1_trained == False, and then set it to True. Otherwise ignore
        lik_1 = GaussianLikelihood()
        GP1 = ExactGPModel(self.train_x,self.train_y,lik_1)
        GP1, lik_1 = train_a_GP(GP1,self.train_x,self.train_y,lik_1,self.training_iter)
        for i in range(self.het_fitting_iter):
            # estimate the noise levels z
            z = torch.log(self.get_r_hat(GP1,lik_1))
            # fit the noise z at train_x
            lik_2 = GaussianLikelihood()
            GP2 = ExactGPModel(self.train_x,z,lik_2)
            GP2, lik_2 = train_a_GP(GP2,self.train_x,z,lik_2,self.training_iter)
            # create a heteroscedastic GP
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                r_pred = lik_2(GP2(self.train_x))
            r = torch.exp(r_pred.mean)
            lik_3 = FixedNoiseGaussianLikelihood(noise=r, learn_additional_noise=False) 
            GP3 = ExactGPModel(self.train_x,self.train_y,lik_3)
            GP3, lik_3 = train_a_GP(GP3,self.train_x,self.train_y,lik_3,self.training_iter)
            GP1 = GP3
            lik_1 = lik_3
            
        self.final_GP = GP3
        self.final_lik = lik_3
        self.final_r = r


    def get_r_hat(self,GP,likelihood):
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            train_pred = likelihood(GP(self.train_x))
        r_hat = torch.sum(0.5*(self.train_y.reshape(1,-1) - train_pred.sample_n(self.var_estimator_n))**2,dim=0)/self.var_estimator_n
        return r_hat