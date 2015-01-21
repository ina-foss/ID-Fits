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


import numpy as np

from utils.file_manager import pickleLoad
from cpp_wrapper.descriptors import LbpDescriptor, Pca, Lda



LBP, ULBP, ULBP_PCA, ULBP_WPCA, ULBP_PCA_LDA, ULBP_PCA_JB = "lbp", "ulbp", "ulbp_pca", "ulbp_wpca", "ulbp_pca_lda", "ulbp_pca_jb"

descriptor_types = [LBP, ULBP, ULBP_PCA, ULBP_WPCA, ULBP_PCA_LDA, ULBP_PCA_JB]



def computeDescriptors(data, descriptor_type = ULBP_WPCA, learned_models_files = {}, normalize = True):

    if descriptor_type not in descriptor_types:
        raise Exception("Descriptor type unknown")

    pca = None
    if "pca" in descriptor_type:
        pca = Pca(filename=learned_models_files["pca"])

    lda = None
    if "lda" in descriptor_type:
        lda = Lda(learned_models_files["lda"])

    descriptor = LbpDescriptor(descriptor_type, pca=pca, lda=lda)

    sample = descriptor.compute(data[0])
    n_samples = data.shape[0]
    n_features = sample.shape[0]
    descriptors = np.empty((n_samples, n_features), dtype=sample.dtype)
    for i in xrange(n_samples):
        descriptors[i] = descriptor.compute(data[i], normalize=normalize)
    
    if "jb" in descriptor_type:
        jb = pickleLoad(learned_models_files["jb"])
        return jb.transform(descriptors)
    else:
        return descriptors


