import sys
import random

from cpp_wrapper.descriptors import Pca
from sklearn.decomposition import PCA



def computeProbabilisticPCA(data, training_set=None, dim=200, samples_nb=None, whiten=False, descriptor=None):

    pca = PCA(dim, whiten=whiten)
    
    if training_set is None:
        training_set = range(len(data))

    samples_indexes = training_set
    if samples_nb is not None:
        samples_indexes = random.sample(training_set, samples_nb)
    
    if descriptor:
        samples = map(descriptor, data[samples_indexes])
    else:
        samples = data[samples_indexes]
    
    pca.fit(samples)
    return pca



def computeProbabilisticPCAs(data, training_sets, dims=200, samples_nb=1000, whiten=False, descriptor=None):

    if type(training_sets[0]) is not list:
        return computeProbabilisticPCA(data, training_sets, dims, samples_nb, whiten)
    
    pcas = []

    for i, training_set in enumerate(training_sets):
        print "Computing PCA #%d" % (i+1)
        pcas.append(computeProbabilisticPCA(data, training_set, dims, samples_nb, whiten, descriptor))
    
    return pcas



    
def computePCA(data, training_set=None, dim=200, samples_nb=None, whiten=False, descriptor=None):

    pca = Pca()
    
    if training_set is None:
        training_set = range(len(data))

    samples_indexes = training_set
    if samples_nb is not None:
        samples_indexes = random.sample(training_set, samples_nb)
    
    if descriptor:
        samples = map(descriptor, data[samples_indexes])
    else:
        samples = data[samples_indexes]
    
    pca.create(samples, dim)
    return pca



def computePCAs(data, training_sets, dims=200, samples_nb=1000, whiten=False, descriptor=None):

    if type(training_sets[0]) is not list:
        return computePCA(data, training_sets, dims, samples_nb, whiten)
    
    pcas = []

    for i, training_set in enumerate(training_sets):
        print "Computing PCA #%d" % (i+1)
        pcas.append(computePCA(data, training_set, dims, samples_nb, whiten, descriptor))
    
    return pcas
