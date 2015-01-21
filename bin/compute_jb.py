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

execfile(os.path.join(os.path.dirname(__file__), "fix_imports.py"))
import config
from datasets import lfw
from learning.joint_bayesian import JointBayesian
from utils.file_manager import pickleSave, makedirsIfNeeded



def computeJointBayesian(data):

    samples_indexes = range(len(data))
    indexes, y = lfw.loadTrainingDataLabels(samples_indexes, min_nb_samples_per_class=10)

    print "Computing Joint Bayesian with %d classes of other 10 samples" % len(set(y))
    jb = JointBayesian()
    jb.fit(data[indexes], y)
        
    return jb



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Computes Joint Bayesian models")
    parser.add_argument("descriptors_file", help="descriptors on which to compute the model (e.g. ulbp_pca_not_normalized_lfwa.npy)")
    parser.add_argument("-o", dest="output_file", default=None, help="where to save JB")
    args = parser.parse_args()
    
    filename = args.descriptors_file.strip()
    print "Using %s descriptors"%filename
    if filename.find("_not_normalized_") < 0:
        raise Exception("Need to use a non normalized descriptor")
    
    basename = os.path.basename(os.path.splitext(filename)[0]).replace("_not_normalized_", "_")
    
    data = np.load(args.descriptors_file)
    jb = computeJointBayesian(data)

    if args.output_file is None:
        filename = os.path.join(config.models_path, "JB.txt")
    else:
        filename = args.output_file.strip()
    makedirsIfNeeded(filename)
    pickleSave(filename, jb)
    print "Results saved in %s" % filename
