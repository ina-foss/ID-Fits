# ID-Fits
# Copyright (c) 2015 Institut National de l'Audiovisuel, INA, All rights reserved.
# 
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 3.0 of the License, or (at your option) any later version.
# 
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
# 
# You should have received a copy of the GNU Lesser General Public
# License along with this library.


import numpy as np
from nose.tools import *
from numpy.testing import *
from Python.learning.gmm import GMM
from sklearn.mixture import GMM as skGMM


class TestGMMOneFeature:
    @classmethod
    def setup_class(cls):
        cls.true_gmm = skGMM(n_components=3)
        cls.true_gmm.weights_ = np.array([0.45, 0.45, 0.1])
        cls.true_gmm.means_ = np.array([[1.0], [2.0], [3.0]])
        cls.true_gmm.covars_ = np.array([[0.25], [0.05], [0.01]])
        cls.true_gmm.converged_ = False

        cls.gmm = GMM(n_components=3, n_init=3)
        cls.gmm.fit(cls.true_gmm.sample(10000).astype(np.float32))
        cls.reindex = np.argsort(cls.gmm.means_, axis=0).ravel()

    
    def test_learning(self):
        assert_allclose(self.gmm.means_[self.reindex], self.true_gmm.means_, atol=0.05)
        assert_allclose(self.gmm.weights_[self.reindex], self.true_gmm.weights_, atol=0.05)
        assert_allclose(self.gmm.covars_[self.reindex], self.true_gmm.covars_, atol=0.05)


    def test_responsabilities_computation(self):
        samples = np.array([[1.0], [2.0], [3.0], [2.5]], dtype=np.float32)
        _, sk_responsabilities = self.true_gmm.score_samples(samples)
        yael_responsabilities = self.gmm.computeResponsabilities(samples)[:, self.reindex]
        assert_allclose(yael_responsabilities, sk_responsabilities, atol=0.1)



class TestGMMTwoFeatures:
    @classmethod
    def setup_class(cls):
        cls.true_gmm = skGMM(n_components=2)
        cls.true_gmm.weights_ = np.array([0.4, 0.6])
        cls.true_gmm.means_ = np.array([[1.0, 1.0], [5.0, 5.0]])
        cls.true_gmm.covars_ = np.array([[0.25, 0.1], [0.05, 0.08]])
        cls.true_gmm.converged_ = False

        cls.gmm = GMM(n_components=2, n_init=3)
        cls.gmm.fit(cls.true_gmm.sample(10000).astype(np.float32))
        cls.reindex = np.argsort(cls.gmm.means_[:, 0], axis=0).ravel()

    
    def test_learning(self):
        assert_allclose(self.gmm.means_[self.reindex], self.true_gmm.means_, atol=0.05)
        assert_allclose(self.gmm.weights_[self.reindex], self.true_gmm.weights_, atol=0.05)
        assert_allclose(self.gmm.covars_[self.reindex], self.true_gmm.covars_, atol=0.05)


    def test_responsabilities_computation(self):
        samples = np.array([[1.0, 1.0], [5.0, 5.0], [7.0, 7.0]], dtype=np.float32)
        _, sk_responsabilities = self.true_gmm.score_samples(samples)
        yael_responsabilities = self.gmm.computeResponsabilities(samples)[:, self.reindex]
        assert_allclose(yael_responsabilities, sk_responsabilities, atol=0.1)
