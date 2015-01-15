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
