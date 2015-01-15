import numpy as np
from nose.tools import *
from numpy.testing import *
from Python.learning.mahalanobis_metric import *


class TestDiagonalMetric:
    
    @classmethod
    def setup_class(cls):
        cls.X = np.array([[-2,-1], [-1, -1], [-1, -2], [0, 1], [1, 0], [1, 1]])
        cls.y = np.array([0, 0, 0, 1, 1, 1])


    def test_complete_learning(self):
        self.diagonal_metric = DiagonalMahalanobisMetric()
        self.diagonal_metric.fit(self.X, self.y)
        d1 = self.diagonal_metric.mesureDistance(self.X[2], self.X[3])
        d2 = self.diagonal_metric.mesureDistance(self.X[0], self.X[1])
        assert_greater(d1, d2)

        
    def test_partial_learning(self):
        self.diagonal_metric = DiagonalMahalanobisMetric()
        self.diagonal_metric.fit(self.X, self.y, n_samples=5)
        d1 = self.diagonal_metric.mesureDistance(self.X[2], self.X[3])
        d2 = self.diagonal_metric.mesureDistance(self.X[0], self.X[1])
        assert_greater(d1, d2)

