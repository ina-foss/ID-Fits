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
