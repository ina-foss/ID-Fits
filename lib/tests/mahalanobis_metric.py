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

