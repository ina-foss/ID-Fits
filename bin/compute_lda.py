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
from learning.lda import computeLDA
from utils.file_manager import pickleSave, makedirsIfNeeded



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Computes LDAs")
    parser.add_argument("descriptors_file", help="descriptors on which to compute LDA (e.g. ulbp_pca_not_normalized_lfwa.npy)")
    args = parser.parse_args()
    
    filename = args.descriptors_file
    print "Using %s descriptors"%filename
    if filename.find("_not_normalized_") < 0:
        raise Exception("Need to use a non normalized descriptor")
    
    basename = os.path.basename(os.path.splitext(filename)[0]).replace("_not_normalized_", "_")
    
    data = np.load(filename)
    lda = computeLDA(data, dim=50)

    filename = os.path.join(config.models_path, "LDA.txt")
    makedirsIfNeeded(filename)
    pickleSave(filename, lda)
    print "Result saved in %s" % filename
