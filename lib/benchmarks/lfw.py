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
import numpy as np

import config
import descriptors
from datasets import lfw



def loadDescriptors(descriptor_type, configuration="test"):

    if configuration == "test":
        sets = lfw.loadTestSets()
    else:
        sets = lfw.loadTrainingSets()

    size = np.sum([len(s) for s in sets])
    descriptors = None
    
    for i, s in enumerate(sets):
        descs = np.load(os.path.join(config.lfw_benchmark_path, configuration, descriptor_type, "set_%d.npy" % (i+1)))

        if descriptors is None:
            descriptors = np.empty((size, descs.shape[1]), dtype=descs.dtype)
        
        for j, index in enumerate(s):
            descriptors[index] = descs[j]

    return descriptors
    


def computeDescriptorsForSets(data, descriptor_type, sets = None, learned_models_dirs = {}, normalize = True):

    descs = []
 
    for i, s in enumerate(sets):

        learned_models_files = {}
        for method, directory in learned_models_dirs.iteritems():
            learned_models_files[method] = os.path.join(directory, "set_%d.txt" % (i+1))

        descs.append(descriptors.computeDescriptors(data[s], descriptor_type, learned_models_files, normalize))

    return descs



def cosineDistance(x, y):
    return np.inner(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


def computeDistanceMatrix(descs, sets_ground_truth, distance=cosineDistance):
    if type(sets_ground_truth) is not list:
        sets_ground_truth = [sets_ground_truth]

    matches_scores = []
    mismatches_scores = []
    
    for matches,mismatches in sets_ground_truth:
        matches_scores.append([])
        mismatches_scores.append([])
        for index in matches:
            matches_scores[-1] += [distance(descs[index], descs[other]) for other in matches[index]]
        for index in mismatches:
            mismatches_scores[-1] += [distance(descs[index], descs[other]) for other in mismatches[index]]
    
    if len(sets_ground_truth) == 1:
        return (matches_scores[0], mismatches_scores[0])
    else:
        return (matches_scores, mismatches_scores)


def statsForThreshold((matches_scores, mismatches_scores), threshold):
    tp = np.sum(matches_scores >= threshold)
    fn = len(matches_scores) - tp
                
    tn = np.sum(mismatches_scores < threshold)
    fp = len(mismatches_scores) - tn
    
    return (fp / float(fp+tn), tp / float(tp+fn))


def computeROC(scores, thresholds=None):
    if thresholds is None:
        score_distribution = np.concatenate(scores)
        thresholds = np.linspace(np.min(score_distribution), np.max(score_distribution), num=100)
    
    fprs,tprs = [],[]
    for threshold in thresholds:
        fpr,tpr = statsForThreshold(scores, threshold)
        fprs.append(fpr)
        tprs.append(tpr)
        
    return fprs,tprs


def computeAccuracy(scores):
    fprs, tprs = computeROC(scores)
    return (np.max(((1-np.asarray(fprs)) + np.asarray(tprs))/2))


def computeMeanAccuracy(sets_scores):
    acc = []
    for scores in zip(*sets_scores):
        acc.append(computeAccuracy(scores))

    return np.mean(acc), np.std(acc, ddof=1) / np.sqrt(len(acc))


def computeMeanROC(sets_scores):
    score_distribution = np.concatenate([np.concatenate(scores) for scores in sets_scores])
    thresholds = np.linspace(np.min(score_distribution), np.max(score_distribution), num=100)


    roc = []
    for scores in zip(*sets_scores):
        roc.append(np.asarray(computeROC(scores, thresholds)))

    roc = np.asarray(roc)
    return (np.mean(roc[:,0,:], axis=0), np.mean(roc[:,1,:], axis=0))
