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
