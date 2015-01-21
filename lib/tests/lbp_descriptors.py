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


import cv2
import numpy as np
from nose.tools import *
from PythonWrapper.descriptors import LbpDescriptor, Pca, Lda


class TestLbpDescriptors:
    @classmethod
    def setup_class(cls):
        cls.image = cv2.imread("tests/lfw_image.jpg", cv2.CV_LOAD_IMAGE_GRAYSCALE)[49:201,84:166]
        cls.pca = Pca(filename="PCA/ulbp_lfwa/PCA_set_1.txt")
        cls.lda = Lda("LDA/ulbp_lfwa/set_1.txt")

    
    def test_lbp(self):
        lbp = LbpDescriptor("lbp")
        desc = lbp.compute(self.image)
        assert_equal(desc.shape[0], (self.image.shape[0]/10)*(self.image.shape[1]/10)*256)


    def test_lbp_step(self):
        lbp = LbpDescriptor("lbp", step=2)
        desc = lbp.compute(self.image)
        assert_equal(desc.shape[0], ((self.image.shape[0]-2-10)/2+1)*((self.image.shape[1]-2-10)/2+1)*256)


    def test_ulbp(self):
        ulbp = LbpDescriptor("ulbp")
        desc = ulbp.compute(self.image)
        assert_equal(desc.shape[0], (self.image.shape[0]/10)*(self.image.shape[1]/10)*59)


    def test_ulbp_pca(self):
        ulbp_pca = LbpDescriptor("ulbp_pca", pca=self.pca)
        desc = ulbp_pca.compute(self.image)
        assert_equal(desc.shape[0], 200)


    def test_ulbp_wpca(self):
        ulbp_pca = LbpDescriptor("ulbp_wpca", pca=self.pca)
        desc = ulbp_pca.compute(self.image)
        assert_equal(desc.shape[0], 200)


    def test_ulbp_pca_lda(self):
        ulbp_pca = LbpDescriptor("ulbp_pca_lda", pca=self.pca, lda=self.lda)
        desc = ulbp_pca.compute(self.image)
        assert_equal(desc.shape[0], 50)
