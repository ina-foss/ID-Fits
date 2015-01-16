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
import nose
from nose.tools import *
from numpy.testing import *
from Python.fisher_vector import *
from Python.utils.file_manager import *
import cPickle as pickle
from yael import *
import importlib


class TestFisherVector(nose.suite.ContextSuite):

    @classmethod
    def setup_class(cls):
        cls.fisher_vector = pickleLoad('fisher_vector.pkl')
        cls.data = np.load('lfw/lfwa.npy')[:5]
        

    def test_yael_computation(self):
        descriptor = computeDenseDescriptor(self.data[0], pca=self.fisher_vector.pca, embed_spatial_information=False)

        print self.fisher_vector.yaelFV(descriptor, improved=False)
        print self.fisher_vector.yaelFV(descriptor, improved=True)
       
        print self.fisher_vector.computeFisherVector(descriptor, improved=True)
        print self.fisher_vector.computeFisherVector(descriptor, improved=False)


if __name__ == '__main__':
    loader = nose.loader.TestLoader()
    loader.loadTestsFromName('TestFisherVector')
    nose.main(config=nose.config.Config(stream=sys.stdout), suite=[loader.loadTestsFromTestClass(TestFisherVector)])
