import os
import sys
import cPickle as pickle
import numpy as np

from sklearn.lda import LDA
from datasets import lfw
from cpp_wrapper.descriptors import Pca


def computeLDA(data, training_sets, dim, pca_dir):
    
    ldas = []

    for i, training_set in enumerate(training_sets):

        descs_id_for_lda, y = lfw.loadTrainingDataLabels(samples_indexes, min_nb_samples_per_class=10)

        print "Computing LDA #%d with %d classes of over 10 samples"%(i+1, len(set(y)))
        samples = data[training_set][descs_id_for_lda]

        pca = Pca(filename=os.path.join(pca_dir, "set_%d.txt"%(i+1)))
        compressed_samples = np.empty((samples.shape[0], pca.eigenvalues.shape[0]), dtype=np.float32)
        for j in range(samples.shape[0]):
            compressed_samples[j] = pca.project(samples[j])
        
        ldas.append(LDA(dim))
        ldas[-1].fit(compressed_samples, y)
        
        del samples, samples_indexes

    return ldas

