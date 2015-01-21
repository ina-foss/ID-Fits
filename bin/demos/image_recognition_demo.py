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
import cv2
import numpy as np

import imp
import os
imp.load_source("fix_imports", os.path.join(os.path.dirname(__file__), os.pardir, "fix_imports.py"))

from database import loadDatabase
from search import *
from descriptors import *

from cpp_wrapper.face_detection import FaceDetector
from cpp_wrapper.alignment import *



def computeDescriptor(descriptor, image):
    detector = FaceDetector()
    alignment = LBFLandmarkDetector(detector="opencv", landmarks=68)
    face_normalization = FaceNormalization()
    face_normalization.setReferenceShape(alignment.getReferenceShape())
    
    #pca_file = "PCA/ulbp_normalized_data_lbf_alignment_68_landmarks_PCA.txt"
    #lda_file = "LDA/ulbp_normalized_data_lbf_alignment_68_landmarks_LDA.txt"
    pca_file = "PCA/ulbp_wlfdb_PCA_200_dim.txt"
    lda_file = "LDA/wlfdb_LDA_50_dim.txt"
    
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
    
    normalized_image = face_normalization.normalize(image, shape)[49:201, 84:166]
    return computeDescriptors([normalized_image], descriptor, pca_file, lda_file)[0]



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Simple demo trying to retrieve the identity of an input image.")
    parser.add_argument("image_file", help="image of the person to identify")
    parser.add_argument("-m", dest="descriptor", default="ulbp_pca_lda", choices=descriptor_types, help="database")
    parser.add_argument("-d", dest="db", default="normalized_data_lbf_alignment_68_landmarks", help="database")
    parser.add_argument("-l", dest="label", default="", help="true label of the input image")
    parser.add_argument("-n", dest="neighbors", type=int, default=50, help="number of neighbors in NN search")
    parser.add_argument("-v", dest="verbose", action="store_true", help="verbose output")
    args = parser.parse_args()

    filename = args.image_file
    nn = args.neighbors
    true_label = args.label
    verbose = args.verbose
    db = args.db
    descriptor = args.descriptor

    database = loadDatabase(desc=descriptor, db=db)

    img = cv2.imread(filename, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    desc = computeDescriptor(descriptor, img)
    
    nn_scores, best_scores = nnSumSearch(desc, database, nn)
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
