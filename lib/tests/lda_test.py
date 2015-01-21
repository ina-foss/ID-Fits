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
from lda import loadLDA
from nose.tools import *
from PythonWrapper.descriptors import Pca, Lda


class TestLda:

    def setup(self):
        self.descriptor = np.load("descriptors/ulbp_not_normalized_lfwa.npy")[0]

    
    def test_lda(self):
        pca = Pca(filename="PCA/ulbp_lfwa/PCA_set_1.txt")
        desc = pca.project(self.descriptor)
        
        sklearn_lda = loadLDA("LDA/ulbp_lfwa/set_1.txt")
        sklearn_lda_desc = sklearn_lda.transform(desc)
        
        lda = Lda("LDA/ulbp_lfwa/set_1.txt")
        lda_desc = lda.project(desc)

        assert_true(np.allclose(sklearn_lda_desc, lda_desc))
