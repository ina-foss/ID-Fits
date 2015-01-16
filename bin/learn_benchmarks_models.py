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
from learning.pca import computePCA
from sklearn.lda import LDA
from learning.joint_bayesian import JointBayesian
from benchmarks import blufr
from datasets import lfw
from utils.file_manager import pickleSave



if __name__ == "__main__":

    # Parse commandline arguments
    parser = argparse.ArgumentParser(description="Computes models for benchmark data")
    parser.add_argument("benchmark", choices=["lfw", "blufr"], help="benchmark")
    parser.add_argument("descriptors_directory", help="which descriptors to use for learning")
    parser.add_argument("model", choices=["pca", "lda", "jb"], help="which model to learn")
    parser.add_argument("dim", type=int, help="number of final dimensions")
    #parser.add_argument("-d", dest="dim_num", type=int, default=200, help="number of dimensions kept by PCA")
    #parser.add_argument("-s", dest="samples_number", type=int, default=2000, help="number of samples used to compute the PCA")
    args = parser.parse_args()


    descriptor_type = os.path.basename(args.descriptors_directory).replace("_not_normalized_", "_")
    if len(descriptor_type) == 0:
        descriptor_type = os.path.basename(os.path.dirname(args.descriptors_directory)).replace("_not_normalized_", "_")
    models_directory = os.path.join(config.benchmarks_path, args.benchmark, "models")

    training_sets_descs = []
    for set_num in range(len(os.listdir(args.descriptors_directory))):
        training_sets_descs.append(np.load(os.path.join(args.descriptors_directory, "set_%d.npy" % (set_num + 1))))

    if args.benchmark == "lfw":
        training_sets = lfw.loadTrainingSets()
    else:
        training_sets = blufr.loadTrainingSets()
        

    for set_num, (training_set, descs) in enumerate(zip(training_sets, training_sets_descs)):
        
        print "\nSet #%d" % (set_num+1)

        if args.model == "pca":
            
            print "Computing PCA..."
            pca_filename = os.path.join(models_directory, "PCA", descriptor_type, "set_%d.txt" % (set_num+1))
            pca = computePCA(descs, dim=args.dim, samples_nb=1000)
            
            if not os.path.exists(os.path.dirname(pca_filename)):
                os.makedirs(os.path.dirname(pca_filename))
            pca.save(pca_filename)
            
        else:

            descs_id, y = lfw.loadTrainingDataLabels(list(training_set), min_nb_samples_per_class=10)

            if args.model == "lda":
                print "Computing LDA..."
                supervised_learning = LDA(args.dim)
                filename = os.path.join(models_directory, "LDA", descriptor_type, "set_%d.txt" % (set_num+1))
            else:
                print "Computing Joint Bayesian..."
                supervised_learning = JointBayesian()
                filename = os.path.join(models_directory, "JB", descriptor_type, "set_%d.txt" % (set_num+1))

            supervised_learning.fit(descs[descs_id], y)
            
            if not os.path.exists(os.path.dirname(filename)):
                os.makedirs(os.path.dirname(filename))
            pickleSave(filename, supervised_learning)
