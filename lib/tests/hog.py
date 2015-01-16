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

from _hog import hog as skimage_hog
from PythonWrapper.descriptors import Hog



class TestHog:
    @classmethod
    def setup_class(cls):
        cls.image = cv2.imread("tests/eye.pgm", cv2.CV_LOAD_IMAGE_GRAYSCALE)


    def test_hog(self):
        skimage_desc = skimage_hog(self.image, orientations=8, pixels_per_cell=(8,8), cells_per_block=(4,4))
        assert skimage_desc.shape[0] == 128

        hog = Hog()
        cpp_desc = hog.compute_(self.image).ravel()
        assert cpp_desc.shape[0] == 128

        print "Squared error between Skimage and own HOG descriptor: %f"%np.linalg.norm(skimage_desc - cpp_desc)**2
        assert np.linalg.norm(skimage_desc - cpp_desc)**2 < 0.5
