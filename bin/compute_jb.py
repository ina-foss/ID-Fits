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
import argparse
import numpy as np

execfile("fix_imports.py")
import config
from datasets import lfw
from learning.joint_bayesian import JointBayesian
from utils.file_manager import pickleSave
from cpp_wrapper.descriptors import Pca



def computeJointBayesian(data, sets, pca_dir):

    jbs = []

    for i in range(len(sets)):
        jbs.append(JointBayesian())
        samples_indexes = []
        for k in range(len(sets)-1):
            samples_indexes += sets[(i+k+1) % len(sets)]

        descs_id_for_lda, y = lfw.loadTrainingDataLabels(samples_indexes, min_nb_samples_per_class=10)

        print "Computing Joint Bayesian #%d with %d classes of other 10 samples" % (i+1, len(set(y)))
        samples = data[samples_indexes][descs_id_for_lda]

        pca = Pca(filename=os.path.join(pca_dir, "set_%d.txt" % (i+1)))
        compressed_samples = np.empty((samples.shape[0], pca.eigenvalues.shape[0]), dtype=np.float32)
        for j in range(samples.shape[0]):
            compressed_samples[j] = pca.project(samples[j])
        
        jbs[-1].fit(compressed_samples, y)
        
    return jbs



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Computes Joint Bayesian models")
    parser.add_argument("descriptors_file", help="descriptors on which to compute the model (e.g. ulbp_pca_not_normalized_lfwa.npy)")
    args = parser.parse_args()
    
    filename = args.descriptors_file
    print "Using %s descriptors"%filename
    if filename.find("_not_normalized_") < 0:
        raise Exception("Need to use a non normalized descriptor")
    
    basename = os.path.basename(os.path.splitext(filename)[0]).replace("_not_normalized_", "_")
    pca_dir = os.path.join(config.models_path, "PCA", basename)
    jb_dir = os.path.join(config.models_path,"JB", basename)
    print "Using PCA files from %s"%pca_dir
    
    sets = lfw.loadSetsPeople()
    data = np.load(filename)
    
    jbs = computeJointBayesian(data, sets, pca_dir)
    
    if not os.path.exists(jb_dir):
        os.makedirs(jb_dir)
    
    for i, jb in enumerate(jbs):
        set_filename = os.path.join(jb_dir, "set_%d.txt"%(i+1))
        pickleSave(set_filename, jb)
    print "Results saved in directory %s"%jb_dir
