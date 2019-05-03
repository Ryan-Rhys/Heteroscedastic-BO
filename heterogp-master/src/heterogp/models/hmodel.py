# Copyright 2018 Joshua G. Albert
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

import numpy as np
import tensorflow as tf

from gpflow.models.model import GPModel
from gpflow import settings


class HGPModel(GPModel):
    """
    A base class for heteroscedastic Gaussian process models, that is, those of the form
    .. math::
       :nowrap:
       \\begin{align}
       \\theta & \sim p(\\theta) \\\\
       f       & \sim \\mathcal{GP}(m(x), k(x, x'; \\theta)) \\\\
       f_i       & = f(x_i) \\\\
       \\phi & \sim p(\\phi) \\\\
       g       & \sim \\mathcal{GP}(m(x), k(x, x'; \\phi)) \\\\
       g_i       & = g(x_i) \\\\
       y_i\,|\,f_i,g_i     & \sim p(y_i|f_i,g_i)
       \\end{align}
    """

    def __init__(self, log_noise_gp, X, Y, kern, likelihood, mean_function, num_samples=1,
                 num_latent=None, name=None):
        super(HGPModel, self).__init__(X, Y, kern, likelihood, mean_function,
                 num_latent=num_latent, name=name)
        self.log_noise_gp = log_noise_gp
        self.num_samples = num_samples

    @autoflow((settings.float_type, [None, None]), (tf.int32,[]))
    def predict_y(self, Xnew, num_samples):
        """
        Compute the mean and variance of held-out data at the points Xnew
        """
        pred_f_mean, pred_f_var = self._build_predict(Xnew)
        noise = tf.exp(self.log_noise_latent.predict_f_samples(Xnew, num_samples))#S, N, num_latent
        variance = tf.square(noise)
        res = tf.map_fn(lambda v: tf.stack(
            self.likelihood.predict_mean_and_var(pred_f_mean, pred_f_var, v),axis=0), variance)
        res = tf.reduce_mean(res,axis=0)
        return tf.unstack(res, axis=0, num=2)

    @autoflow((settings.float_type, [None, None]), (settings.float_type, [None, None]), (tf.int32,[]))
    def predict_density(self, Xnew, Ynew, num_samples):
        """
        Compute the (log) density of the data Ynew at the points Xnew
        Note that this computes the log density of the data individually,
        ignoring correlations between them. The result is a matrix the same
        shape as Ynew containing the log densities.
        """
        pred_f_mean, pred_f_var = self._build_predict(Xnew)
        noise = tf.exp(self.log_noise_latent.predict_f_samples(Xnew, num_samples))#S, N, num_latent
        variance = tf.square(noise)
        res = tf.map_fn(lambda v: self.likelihood.predict_density(pred_f_mean, pred_f_var, Ynew, v), variance)
        return tf.reduce_mean(res,axis=0)
