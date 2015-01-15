import numpy as np
from nose.tools import *
from numpy.testing import *
from Python.learning.large_margin_dimension_reduction import *


class TestLargeMarginDimensionReduction:
    
    @classmethod
    def setup_class(cls):
        cls.X = np.array([[-1, -2], [-2,-1], [-1, -1], [0, 1], [1, 0], [1, 1]])
        cls.y = np.array([0, 0, 0, 1, 1, 1])


    def setup(self):
        self.dimension_reduction = LargeMarginDimensionReduction(n_components=1)
        self.dimension_reduction.W_ = np.array([0.5, 0.5])
        self.dimension_reduction.b_ = 1.25


    def test_learning(self):
        self.dimension_reduction = LargeMarginDimensionReduction(n_components=1, n_iter=1000)
        self.dimension_reduction.fit(self.X, self.y)
        assert_less(abs(np.inner(self.dimension_reduction.W_, np.array([1, -1]))), 0.05)


    def test_mahalanobis_distance(self):
        d1 = self.dimension_reduction.mahalanobisDistance(self.X[2], self.X[3], self.dimension_reduction.W_)
        d2 = self.dimension_reduction.mahalanobisDistance(self.X[0], self.X[1], self.dimension_reduction.W_)
        assert_greater(d1, d2)


    def test_transform(self):
        X = self.dimension_reduction.transform(self.X)
        d1 = np.inner(X[2]-X[3], X[2]-X[3])
        d2 = np.inner(X[0]-X[1], X[0]-X[1])
        assert_equal(X.reshape(self.X.shape[0], X.shape[0]/self.X.shape[0]).shape[1], self.dimension_reduction.n_components_)
        assert_greater(d1, d2)
