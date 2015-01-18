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
from datasets import lfw
from descriptors import descriptor_types, computeDescriptors
from utils.file_manager import makedirsIfNeeded



if __name__ == "__main__":
    
    # Parse commandline arguments
    parser = argparse.ArgumentParser(description="Computes descriptors")
    parser.add_argument("descriptor_type", choices=descriptor_types)
    parser.add_argument("dataset", help="dataset to use (e.g. lfwa.npy)")
    parser.add_argument("output_file", help="where to write results")
    parser.add_argument("-c", dest="cropped", action="store_false", help="crop the images (set true by default)")
    parser.add_argument("-n", dest="normalize", action="store_true", help="normalize the descriptors")
    parser.add_argument("-p", dest="pca_file", default=None, help="PCA file to use")
    parser.add_argument("-l", dest="lda_file", default=None, help="LDA file to use")
    parser.add_argument("-j", dest="jb_file", default=None, help="Joint Bayesian file to use")
    args = parser.parse_args()

    descriptor_type = args.descriptor_type
    output_file = args.output_file.strip()
    normalize = args.normalize

    # Load data
    data = np.load(args.dataset)
    if args.cropped:
        data = lfw.preprocessData(data)
    
    # Compute descriptors
    learned_models_files = {}
    complete_learned_models_files = {
        "pca": args.pca_file,
        "lda": args.lda_file,
        "jb": args.jb_file
    }
    for name, filename in complete_learned_models_files.iteritems():
        if filename is not None:
            learned_models_files[name] = filename.strip()
        elif name in descriptor_type:
            raise Exception("Need to pass the model for %s" % name.upper())
    descs = computeDescriptors(data, descriptor_type, learned_models_files, normalize)


    # Save results
    makedirsIfNeeded(output_file)
    np.save(output_file, descs)
    print "Results saved in %s" % output_file
