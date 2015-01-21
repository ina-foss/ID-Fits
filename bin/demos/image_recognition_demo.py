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
import cv2
import numpy as np

execfile("fix_imports.py")

import config
from database import loadDatabase
from search import *
from utils.file_manager import pickleLoad
from learning.joint_bayesian import jointBayesianDistance

from cpp_wrapper.face_detection import FaceDetector
from cpp_wrapper.alignment import *
from cpp_wrapper.descriptors import *



def computeDescriptor(descriptor_type, database, image):
    detector = FaceDetector()
    alignment = LBFLandmarkDetector(detector="opencv", landmarks=68)
    face_normalization = FaceNormalization()
    face_normalization.setReferenceShape(alignment.getReferenceShape())
    pca = Pca(filename=os.path.join(config.models_path, "PCA_%s.txt" % database))
    lda = Lda(os.path.join(config.models_path, "LDA_%s.txt" % database))
    jb = pickleLoad(os.path.join(config.models_path, "JB_%s.txt" % database))
    descriptor = LbpDescriptor(descriptor_type, pca=pca, lda=lda)
    
    face = detector.detectFaces(image)
    if len(face) == 0:
        raise Exception("No faces detected")
    face = face[0]
    shape = alignment.detectLandmarks(image, face)

    copy = np.copy(image)
    cv2.rectangle(copy, face[:2], face[2:], (0,0,255), 1)
    for landmark in shape:
        cv2.circle(copy, tuple(landmark.astype(np.int)), 1, (0,255,0), -1)
    cv2.imshow("Face detection and alignment", copy)
    
    face_normalization.normalize(image, shape)
    image = image[49:201, 84:166]
    
    if "jb" in descriptor_type:
        desc = descriptor.compute(image, normalize=False)
        return jb.transform(desc[np.newaxis]).ravel()
    else:
        return descriptor.compute(image)



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Simple demo trying to retrieve the identity of an input image.")
    parser.add_argument("image_file", help="image of the person to identify")
    parser.add_argument("-m", dest="descriptor", default="ulbp_pca_jb", help="descriptor")
    parser.add_argument("-d", dest="database", default="lfw_normalized_lbf_68_landmarks", help="database")
    parser.add_argument("-l", dest="label", default="", help="true label of the input image")
    parser.add_argument("-n", dest="neighbors", type=int, default=50, help="number of neighbors in NN search")
    parser.add_argument("-v", dest="verbose", action="store_true", help="verbose output")
    args = parser.parse_args()

    filename = args.image_file
    nn = args.neighbors
    true_label = args.label
    verbose = args.verbose
    database_name = args.database
    descriptor_type = args.descriptor

    database = loadDatabase(desc=descriptor_type, db=database_name)

    img = cv2.imread(filename, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    desc = computeDescriptor(descriptor_type, database_name, img)

    if "jb" in descriptor_type:
        similarity = jointBayesianDistance
    else:
        similarity = np.inner
    nn_scores, best_scores = nnSearch(desc, database, nn, similarity=similarity)
    #nn_scores, best_scores = nnGaussianKernelSearch(desc, database, nn)
    output, max_number_nn = nn_scores[0]
    
    if len(true_label) > 0:
        true_match_scores = [(label,score) for label, score in best_scores if label == true_label][:10]
    
    print "Result: %s with %d %d-NN"%(output, max_number_nn, nn)
    if output != true_label and len(true_label) > 0:
        scores_dict = dict(nn_scores)
        if true_label in scores_dict:
            true_label_nn = scores_dict[true_label]
        else:
            true_label_nn = 0
        print "%d %d-NN with %s as label"%(true_label_nn, nn, true_label)
    print ""

    print "Ranking:"
    for label, score in nn_scores[:10]:
        print label, "\t", score
    print ""
    
    if not verbose:
        best_scores = best_scores[:10]
    print "%d NN scores:"%len(best_scores)
    for label, score in best_scores:
        print label, "\t", score

    if len(true_label) > 0:
        print "\n", "%d best true matches' scores:"%len(true_match_scores)
        for label, score in true_match_scores:
            print label, "\t", score
    
    cv2.waitKey(0)
