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
from learning.pca import *
from utils.file_manager import makedirsIfNeeded



def compute(filename, dims, samples_nb, output_file=None):
    if filename.find("_not_normalized_") < 0:
        raise Exception("Need to use a non normalized descriptor")
    if samples_nb <= 0:
        samples_nb = None
        
    print "Using %s to compute PCA" % filename

    data = np.load(filename)
    pca = computePCA(data, dim=dims, samples_nb=samples_nb)

    if output_file is None:
        output_file = os.path.join(config.models_path, "PCA.txt")
    makedirsIfNeeded(output_file)
    pca.save(output_file)
    print "Results saved in %s" % output_file




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Computes PCA")
    parser.add_argument("input_data_file", help="descriptor file on which to compute PCA (e.g. ulbp_not_normalized_lfwa.npy)")
    parser.add_argument("-d", dest="dim_num", type=int, default=200, help="number of dimensions kept by PCA")
    parser.add_argument("-s", dest="samples_number", type=int, default=1000, help="number of samples used to compute the PCA")
    parser.add_argument("-o", dest="output_file", default=None, help="where to save PCA")
    args = parser.parse_args()

    filename = args.input_data_file.strip()
    dims = args.dim_num
    samples_nb = args.samples_number
    output_file = args.output_file.strip()

    
    if filename.find("_not_normalized_") < 0:
        raise Exception("Need to use a non normalized descriptor")
    if samples_nb <= 0:
        samples_nb = None
        
    print "Using %s to compute PCA" % filename

    data = np.load(filename)
    pca = computePCA(data, dim=dims, samples_nb=samples_nb)

    if output_file is None:
        output_file = os.path.join(config.models_path, "PCA.txt")
    makedirsIfNeeded(output_file)
    pca.save(output_file)
    print "Results saved in %s" % output_file
