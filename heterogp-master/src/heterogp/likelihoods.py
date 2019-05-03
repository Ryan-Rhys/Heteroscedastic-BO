# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import absolute_import

import numpy as np
import tensorflow as tf

from gpflow import densities
from gpflow import settings
from gpflow.decors import autoflow
from gpflow.decors import params_as_tensors
from gpflow.likelihoods import Likelihood
from gpflow.quadrature import hermgauss
from .latent import Latent


class HeteroscedasticLikelihood(Likelihood):
    """
    Represents a generalization of the likelihood to 
    allow input-dependent error, which means each function 
    now accepts more arguments.
    
    The heteroscedastic error should always be the same
    shape as the data vector.
    """
    def __init__(self, log_noise_latent, min_noise=1e-2, name=None):
        super().__init__(name=name)
        self.log_noise_latent = log_noise_latent
        self.min_noise = tf.convert_to_tensor(min_noise, dtype=settings.float_type)
        assert isinstance(log_noise_latent, Latent)
        
    @params_as_tensors
    def hetero_noise(self, X):
        """
        Calculates the heteroscedastic variance at points X.
        X must be of shape [S, N, D]
        Returns [S, N, num_latent]
        """
        log_noise, _ ,_ = self.log_noise_latent.sample_from_conditional(X, full_cov=True)
        hetero_noise = tf.exp(log_noise)
        hetero_noise = tf.where(hetero_noise < self.min_noise, 
                                tf.fill(tf.shape(hetero_noise), self.min_noise), 
                                hetero_noise)
        return hetero_noise
    
    @autoflow((settings.float_type, [None, None]), (tf.int32, []))
    def compute_hetero_noise(self, X, num_samples):
        """
        Computes the hetero_noise at X by sampling `num_sample` times.
        X is shape [N,D]
        """
        X = tf.tile(X[None,:,:], (num_samples,1,1))
        return self.hetero_noise(X) 
        
    @params_as_tensors
    def logp(self, F, Y, hetero_variance, *args):
        """The log-likelihood function."""
        raise NotImplemented("sub class must...")

    @params_as_tensors
    def conditional_mean(self, F, hetero_variance, *args):  # pylint: disable=R0201
        """The mean of the likelihood conditioned on latent."""
        raise NotImplemented("sub class must...")

    @params_as_tensors
    def conditional_variance(self, F, hetero_variance, *args): # pylint: disable=R0201
        """The var of the likelihood conditioned on latent."""
        raise NotImplemented("sub class must...")

    def predict_mean_and_var(self, Fmu, Fvar, hetero_variance, *args):
        """
        Given a Normal distribution for the latent function,
        return the mean of Y
        if
            q(f) = N(Fmu, Fvar)
        and this object represents
            p(y|f)
        then this method computes the predictive mean
           \int\int y p(y|f)q(f) df dy
        and the predictive variance
           \int\int y^2 p(y|f)q(f) df dy  - [ \int\int y p(y|f)q(f) df dy ]^2
        Here, we implement a default Gauss-Hermite quadrature routine, but some
        likelihoods (e.g. Gaussian) will implement specific cases.
        """
        gh_x, gh_w = hermgauss(self.num_gauss_hermite_points)
        gh_w /= np.sqrt(np.pi)
        gh_w = gh_w.reshape(-1, 1)
        shape = tf.shape(Fmu)
        Fmu, Fvar, hetero_variance = [tf.reshape(e, (-1, 1)) for e in (Fmu, Fvar, hetero_variance)]
        # each element of arg must have same number of latent
        # assumes heteroscedastic error has same shape as Y
        # TODO  tile to match latent if not already
        args = [tf.reshape(e, (-1, 1)) if isinstance(e, tf.Tensor) else e for e in args]
        X = gh_x[None, :] * tf.sqrt(2.0 * Fvar) + Fmu

        # here's the quadrature for the mean
        conditional_mean = self.conditional_mean(X, hetero_variance, *args)
        E_y = tf.reshape(tf.matmul(conditional_mean, gh_w), shape)

        # here's the quadrature for the variance
        integrand = self.conditional_variance(X, hetero_variance, *args) + tf.square(conditional_mean)
        V_y = tf.reshape(tf.matmul(integrand, gh_w), shape) - tf.square(E_y)

        return E_y, V_y

    def predict_density(self, Fmu, Fvar, Y, hetero_variance, *args):
        """
        Given a Normal distribution for the latent function, and a datum Y,
        compute the (log) predictive density of Y.
        i.e. if
            q(f) = N(Fmu, Fvar)
        and this object represents
            p(y|f)
        then this method computes the predictive density
           \int p(y=Y|f)q(f) df
        Here, we implement a default Gauss-Hermite quadrature routine, but some
        likelihoods (Gaussian, Poisson) will implement specific cases.
        """
        gh_x, gh_w = hermgauss(self.num_gauss_hermite_points)
        gh_w = gh_w.reshape(-1, 1) / np.sqrt(np.pi)
        shape = tf.shape(Fmu)
        Fmu, Fvar, Y, hetero_variance = [tf.reshape(e, (-1, 1)) for e in (Fmu, Fvar, Y, hetero_variance)]
        args = [tf.reshape(e, (-1, 1)) if isinstance(e, tf.Tensor) else e for e in args]
        X = gh_x[None, :] * tf.sqrt(2.0 * Fvar) + Fmu

        Y = tf.tile(Y, [1, self.num_gauss_hermite_points])  # broadcast Y to match X

        logp = self.logp(X, Y, hetero_variance, *args)

        return tf.reshape(tf.log(tf.matmul(tf.exp(logp), gh_w)), shape)

    @params_as_tensors
    def variational_expectations(self, Fmu, Fvar, Y, hetero_variance, *args):
        """
        Compute the expected log density of the data, given a Gaussian
        distribution for the function values.
        if
            q(f) = N(Fmu, Fvar)
        and this object represents
            p(y|f)
        then this method computes
           \int (\log p(y|f)) q(f) df.
        Here, we implement a default Gauss-Hermite quadrature routine, but some
        likelihoods (Gaussian, Poisson) will implement specific cases.
        """
        gh_x, gh_w = hermgauss(self.num_gauss_hermite_points)
        gh_x = gh_x.reshape(1, -1)
        gh_w = gh_w.reshape(-1, 1) / np.sqrt(np.pi)
        shape = tf.shape(Fmu)
        Fmu, Fvar, Y,hetero_variance = [tf.reshape(e, (-1, 1)) for e in (Fmu, Fvar, Y, hetero_variance)]
        args = [tf.reshape(e, (-1, 1)) if isinstance(e,tf.Tensor) else e for e in args]
        X = gh_x * tf.sqrt(2.0 * Fvar) + Fmu
        Y = tf.tile(Y, [1, self.num_gauss_hermite_points])  # broadcast Y to match X
        logp = self.logp(X, Y, hetero_variance, *args)
        return tf.reshape(tf.matmul(logp, gh_w), shape)


class HeteroscedasticGaussian(HeteroscedasticLikelihood):
    def __init__(self, log_noise_latent, name=None):
        super().__init__(log_noise_latent, name=name)
        
    @params_as_tensors
    def logp(self, F, Y, hetero_variance, *unused_args):
        """The log-likelihood function."""
        return densities.gaussian(F, Y, hetero_variance)

    @params_as_tensors
    def conditional_mean(self, F, unused_hetero_variance, *unused_args):  # pylint: disable=R0201
        """The mean of the likelihood conditioned on latent."""
        return F

    @params_as_tensors
    def conditional_variance(self, F, hetero_variance, *unused_args):
        return hetero_variance
