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


import sys
import cv2
import random
import argparse
import numpy as np
from utils.file_manager import pickleLoad, pickleSave
from utils.parallelize import parallelize
from learning.gmm import GMM
from dataset import *
from PythonWrapper.descriptors import *
from sklearn.decomposition import PCA
from yael import yael



def computeDenseDescriptor(image, cell_size=24, step=2, scales=5, scale_factor=1.41, pca=None, embed_spatial_information=False):
    descriptor = LbpDescriptor('ulbp', cell_size=cell_size, step=step)
    ndims = 59
        
    descs = np.empty((0, ndims), dtype=np.float32)
    img = image[44:206,62:188]
    for s in range(scales):
        patches = descriptor.compute(img, normalize=False, flatten=False)
        descs = np.append(descs, patches, axis=0)
        
        """
        if embed_spatial_information:
            new_desc = np.empty((desc.shape[0]+2), dtype=np.float32)
            new_desc[:-2] = desc
            new_desc[-2:] = np.array([i/float(img.shape[0])-0.5, j/float(img.shape[0])-0.5])
            desc = new_desc
        """
                    
        img = cv2.resize(img, None, fx=1/scale_factor, fy=1/scale_factor, interpolation=cv2.INTER_NEAREST)

    if pca is not None:
        return pca.transform(descs)
    else:
        return descs


class FisherVectors:

    def __init__(self, cell_size=24, step=2, scales=3, embed_spatial_information=False, filename=None):
        if filename:
            self = pickleLoad(filename)
        else:
            self.cell_size = cell_size
            self.step = step
            self.scales = scales
            self.embed_spatial_information = embed_spatial_information


    def computePcaOnLocalDescriptors(self, images, n_image_samples=500, n_pca_components=None):
	n_patches, n_features = computeDenseDescriptor(images[0]).shape
        random_indexes = random.sample(range(len(images)), n_image_samples)
    
        print 'Computing descriptors for PCA'
        sys.stdout.flush()
        pca_descs = np.empty((n_image_samples*n_patches, n_features), dtype=np.float32)
        for i, image in enumerate(images[random_indexes]):
            pca_descs[i*n_patches:(i+1)*n_patches] = computeDenseDescriptor(image, scales=5)
    
        print 'Computing PCA'
        sys.stdout.flush()
        self.pca = PCA(n_components=n_pca_components, copy=False)
        self.pca.fit(pca_descs)

        print 'PCA computation done'
        sys.stdout.flush()


    def computeGMM(self, images, n_image_samples=None, n_threads=8):
        if not n_image_samples or n_image_samples <= 0:
            n_image_samples = images.shape[0]
            random_indexes = range(n_image_samples)
        else:
            random_indexes = random.sample(range(len(images)), n_image_samples)
        
        print 'Computing descriptors for GMM'
        sys.stdout.flush()
        n_patches = computeDenseDescriptor(images[0]).shape[0]
        gmm_descs_filename = "cache/fisher_vectors/gmm_descs.mmap"
        descriptors = parallelize(_parallelDenseDescriptorComputation, images[random_indexes], (n_image_samples*n_patches, self.pca.n_components_), np.float32, args=[n_patches, self.pca], output_file=gmm_descs_filename, n_jobs=n_threads, load_as_array=True)

        print 'Computing GMM'
        sys.stdout.flush()
        self.gmm = GMM(n_components=512, n_threads=n_threads)
        self.gmm.fit(descriptors)

        print 'GMM computation done'
        sys.stdout.flush()
    

    def computeFisherVector(self, patches, improved=True):
        K = self.gmm.n_components
        N, d = patches.shape

        vector = np.empty((2*K, d), dtype=np.float32)
    
        soft_assignments = self.gmm.computeResponsabilities(patches)
        squared_patches = patches ** 2
    
        for k in range(K):
            S_0 = soft_assignments[:,k].mean()
            S_1 = (soft_assignments[:,k,np.newaxis] * patches).mean(axis=0)
            S_2 = (soft_assignments[:,k,np.newaxis] * squared_patches).mean(axis=0)
        
            vector[k] = (S_1 - self.gmm.means_[k]*S_0) / (np.sqrt(self.gmm.weights_[k] * self.gmm.covars_[k]))
            vector[K+k] = (S_2 - 2*self.gmm.means_[k]*S_1 + (self.gmm.means_[k]**2-self.gmm.covars_[k]**2)*S_0) / (np.sqrt(2*self.gmm.weights_[k]) * self.gmm.covars_[k])
    
        vector = vector.ravel()
    
        if improved:
            # Signed square-rooting
            vector = np.sign(vector) * np.sqrt(np.abs(vector))
            
            # L2 normalization
            vector /= np.linalg.norm(vector)
        
        return vector


    def yaelFV(self, patches, improved=True):
        K = self.gmm.n_components
        N, d = patches.shape
        
        flags = yael.GMM_FLAGS_MU | yael.GMM_FLAGS_SIGMA
        v = yael.numpy_to_fvec(patches)
        out = yael.fvec_new_0(2*K*d)
        self.gmm.initYaelGmm()
        yael.gmm_fisher(patches.shape[0], v, self.gmm.yael_gmm, flags, out)
        vector = yael.fvec_to_numpy_acquire(out, 2*K*d)

        if improved:
            # Signed square-rooting
            vector = np.sign(vector) * np.sqrt(np.abs(vector))
            
            # L2 normalization
            vector /= np.linalg.norm(vector)
        
        return vector
        


    
def _parallelDenseDescriptorComputation(data, output, i, n_patches, pca=None, embed_spatial_information=False):
    output[i*n_patches:(i+1)*n_patches] = computeDenseDescriptor(data[i], pca=pca, embed_spatial_information=embed_spatial_information)
            


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Perform Fisher Vectors learning and computation')
    parser.add_argument('command', choices=['pca_learning', 'gmm_learning'], help='command to execute')
    #parser.add_argument('data', help='data to use for computations')
    parser.add_argument('-i', dest='input_file', default='fisher_vector.pkl', help='previously learnt (or partially learnt) fisher vector models')
    parser.add_argument('-o', dest='output_file', default='fisher_vector.pkl', help='where to write computations results')
    parser.add_argument('-j', dest='n_threads', type=int, default=1, help='number of threads to use')
    args = parser.parse_args()

    base_path = "/rex/store1/home/tlorieul/"
    training_set = loadDevData(filename=(base_path + 'lfw/peopleDevTrain.txt'), mapping_filename=(base_path + 'lfw/mapping.txt'))
    data = np.load(base_path + 'lfw/lfwa.npy')
    training_data = data[training_set]

    
    if args.command == 'pca_learning':
        fisher_vectors = FisherVectors(scales=5)
        fisher_vectors.computePcaOnLocalDescriptors(training_data, n_pca_components=20)
        pickleSave(args.output_file, fisher_vectors)

    elif args.command == 'gmm_learning':
        fisher_vectors = pickleLoad(args.input_file)
        fisher_vectors.computeGMM(training_data, n_threads=args.n_threads)
        pickleSave(args.output_file, fisher_vectors)
