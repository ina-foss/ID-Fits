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


import argparse
import numpy as np
from dataset import loadMapping, loadSetsPeople, preprocessData
from descriptors import *
from lda import loadLDA


def loadDatabase(desc="ulbp_wpca", db="lfwa"):
    npzfile = np.load("db/" + desc + "_" + db + ".npz")
    return npzfile["descriptors"], npzfile["labels"]


if __name__ == "__main__":

    datasets = ["lfwa", "normalized_data_lbf_alignment_51_landmarks", "normalized_data_lbf_alignment_68_landmarks"]

    parser = argparse.ArgumentParser(description="Builds database for IR.")
    parser.add_argument("dataset", choices=datasets, help="which dataset to use to build the database")
    parser.add_argument("-d", dest="descriptor", default="ulbp_pca_lda", choices=descriptor_types, help="descriptor to compute")
    args = parser.parse_args()

    dataset = args.dataset
    #pca_file = "PCA/ulbp_" + dataset + "_PCA.txt"
    #lda_file = "LDA/ulbp_" + dataset + "_LDA.txt"
    pca_file = "PCA/ulbp_wlfdb_PCA_200_dim.txt"
    lda_file = "LDA/wlfdb_LDA_50_dim.txt"
    descriptor = args.descriptor
    filename = descriptor + "_" + dataset

    mapping = loadMapping()
    labels = [""]*len(mapping)

    sets = loadSetsPeople()
    test_set = []
    for s in sets[1:]:
        test_set += s
    
    data = np.load("lfw/" + dataset + ".npy")[test_set]
    data = preprocessData(data)

    descs = np.asarray(computeDescriptors(data, descriptor, pca_file, lda_file))
    
    for name, image_id in mapping:
        index = mapping[(name, image_id)]
        if index in test_set:
            labels[index] = name
        else:
            print name
    labels = np.asarray(labels)[test_set]

    output_file = "db/" + filename
    np.savez(output_file, descriptors=descs, labels=labels)
    print "Saved to " + output_file + ".npz"

