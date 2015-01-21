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
import time
import argparse
import numpy as np

execfile(os.path.join(os.path.dirname(__file__), "fix_imports.py"))
import config
from descriptors import descriptor_types
from datasets import lfw
from benchmarks import lfw as lfw_bench
from benchmarks import blufr




if __name__ == "__main__":

    # Parse commandline arguments
    parser = argparse.ArgumentParser(description="Computes descriptors for LFW benchmark")
    parser.add_argument("benchmark", choices=["lfw", "blufr"], help="benchmark")
    parser.add_argument("configuration", choices=["training", "test"], help="which use for the descriptors")
    parser.add_argument("descriptor_type", choices=descriptor_types)
    parser.add_argument("dataset", help="dataset to use (e.g. lfwa.npy)")
    parser.add_argument("-c", dest="cropped", action="store_false", help="crop the images (set true by default)")
    parser.add_argument("-n", dest="normalize", action="store_true", help="normalize the descriptors")
    args = parser.parse_args()

    descriptor_type = args.descriptor_type
    dataset_file = args.dataset
    normalize = args.normalize


    # Load data
    dataset = os.path.splitext(os.path.basename(dataset_file))[0]
    data = np.load(dataset_file)
    if args.cropped:
        data = lfw.preprocessData(data)
    sub_parts = descriptor_type.split("_")[:-1]


    # Compute descriptors
    print "Computing descriptors..."

    if args.benchmark == "lfw":
        if args.configuration == "training":
            sets = lfw.loadTrainingSets()
        else:
            sets = lfw.loadTestSets()
        models_directory = os.path.join(config.lfw_benchmark_path, "models")
    else:
        if args.configuration == "training":
            sets = blufr.loadTrainingSets()
        else:
            sets = blufr.loadTestSets()
        models_directory = os.path.join(config.blufr_benchmark_path, "models")
    learned_models_dirs = {}

    models_path = os.path.join(config.lfw_benchmark_path, "models")

    if "pca" in descriptor_type:
        learned_models_dirs["pca"] = os.path.join(models_path, "PCA", "_".join(sub_parts[:1]) + "_" + dataset)
        print "Using PCA files from directory %s" % learned_models_dirs["pca"]

    if "lda" in descriptor_type:
        learned_models_dirs["lda"] = os.path.join(models_path, "LDA", "_".join(sub_parts[:2]) + "_" + dataset)
        print "Using LDA files from directory %s" % learned_models_dirs["lda"]

    if "jb" in descriptor_type:
        learned_models_dirs["jb"] = os.path.join(models_path, "JB", "_".join(sub_parts[:2]) + "_" + dataset)
        print "Using Joint Bayesian model files from directory %s" % learned_models_dirs["jb"]

    t = time.time()

    descs = lfw_bench.computeDescriptorsForSets(np.asarray(data), descriptor_type=descriptor_type, sets=sets, learned_models_dirs=learned_models_dirs, normalize=normalize)
    
    print "Done in %.2f seconds"%(time.time()-t)


    # Save results
    descriptors_path = os.path.join(config.benchmarks_path, args.benchmark, args.configuration)
    
    if normalize:
        dirname = os.path.join(descriptors_path, descriptor_type + "_" + dataset)
    else:
        dirname = os.path.join(descriptors_path, descriptor_type + "_not_normalized_" + dataset)

    if not os.path.exists(dirname):
        os.makedirs(dirname)

    for i, desc in enumerate(descs):
        np.save(os.path.join(dirname, "set_%d" % (i+1)), desc)
    print "Results saved in %s"% dirname
