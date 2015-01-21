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


import os
import sys
import cPickle as pickle
import numpy as np

from sklearn.lda import LDA
from datasets import lfw
from cpp_wrapper.descriptors import Pca


def computeLDA(data, dim):
    samples_indexes = range(len(data))
    indexes, y = lfw.loadTrainingDataLabels(samples_indexes, min_nb_samples_per_class=10)

    samples = data[indexes]
    lda = LDA(dim)
    lda.fit(data[indexes], y)

    return lda


def computeLDAs(data, training_sets, dim, pca_dir):
    
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

