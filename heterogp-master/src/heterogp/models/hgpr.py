# Copyright 2016 James Hensman, Valentine Svensson, alexggmatthews, fujiisoup
#
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

import tensorflow as tf
import numpy as np

from .hmodel import HGPModel
from gpflow import settings


class HGPR(HGPModel):
    """
    Gaussian Process Regression.
    This is a vanilla implementation of GP regression with a Gaussian
    likelihood.  Multiple columns of Y are treated independently.
    The log likelihood in this models is sometimes referred to as the 'marginal log likelihood', and is given by
    .. math::
       \\log p(\\mathbf y \\,|\\, \\mathbf f) = \\mathcal N\\left(\\mathbf y\,|\, 0, \\mathbf K + \\sigma_n \\mathbf I\\right)
    """
    def __init__(self, log_noise_gp, X, Y, kern, mean_function, **kwargs):
        """
        X is a data matrix, size N x D
        Y is a data matrix, size N x R
        kern, mean_function are appropriate GPflow objects
        """
        super(HGPR, self).__init__(log_noise_gp, X, Y, kern, None, mean_function, **kwargs)

    @name_scope('likelihood')
    @params_as_tensors
    def _build_likelihood(self):
        """
        Construct a tensorflow function to compute the likelihood.
            \log p(Y | theta).
            This is,
            p(Y | f, g) p(f, g)
            where P(Y | f,g) = N[Y | f, g**2)
            P(f,g) = N[f | mu_f, K_f] N[g | mu_g, K_g]
            P(Y) = int df int dg N[Y | f, g**2) N[f | mu_f, K_f] N[g | mu_g, K_g]
            P(Y) = E_g[ int df N[Y | f, g**2) N[f | mu_f, K_f] ] 
            with g ~ N[g | mu_g, K_g]
            P(Y) = E_g[ N[Y | mu_f, K_f + g**2] ]
            0.5*N*log(2pi) - log|L| - (Y - m)(K_f + g**2)(Y - m)
        """
        noise = tf.exp(self.log_noise_gp.predict_f_samples(self.X, self.num_samples))#S, N, num_latent
        variance = tf.transpose(tf.square(noise), (0,2,1))# S, num_latent, N
        

        K = self.kern.K(self.X)# N, N
        K = tf.tile(K[None, None, :,:], (self.num_samples, self.num_latent, 1, 1))#S, num_latent, N, N
        K = K + tf.matrix_diag(variance)
        L = tf.cholesky(K)# S, num_latent, N, N
        m = self.mean_function(self.X)# N, 1
        d = self.Y - m # N, num_latent
        d = tf.tile(d[None, :,:], (self.num_samples, 1,1))# S, N, num_latent
        d = tf.transpose(d,(0,2,1))[..., None]# S, num_latent, N, 1
        alpha = tf.matrix_triangular_solve(L, d, lower=True)#S, num_latent, N, 1
        num_dims = tf.cast(tf.shape(self.X)[0], L.dtype)
        p = -0.5 * tf.reduce_sum(tf.square(alpha), axis=[2,3])#S, num_latent
        p -= 0.5 * num_dims * np.log(2 * np.pi)
        p -= tf.reduce_sum(tf.log(tf.matrix_diag_part(L)), axis=2)#S, num_latent
        p = tf.reduce_sum(p, axis=1)#S
        return tf.reduce_mean(p)

    @name_scope('predict')
    @params_as_tensors
    def _build_predict(self, Xnew, full_cov=False):
        """
        Xnew is a data matrix, point at which we want to predict
        This method computes
            p(F* | Y ) = N[F* | (K - K g**-2 K / (1 - g^-1 K g^-1)) (Y / g**2 + K^{-1} mu), K - K g**-2 K / (1 - g^-1 K g^-1)]
        where F* are points on the GP at Xnew, Y are noisy observations at X.
        """
        y = self.Y - self.mean_function(self.X)
        Kmn = self.kern.K(self.X, Xnew)
        Kmm_sigma = self.kern.K(self.X) + tf.eye(tf.shape(self.X)[0], dtype=settings.float_type) * self.likelihood.variance
        Knn = self.kern.K(Xnew) if full_cov else self.kern.Kdiag(Xnew)
        f_mean, f_var = base_conditional(Kmn, Kmm_sigma, Knn, y, full_cov=full_cov, white=False)  # N x P, N x P or P x N x N
        return f_mean + self.mean_function(Xnew), f_var
