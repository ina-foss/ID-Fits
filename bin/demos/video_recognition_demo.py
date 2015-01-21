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
from matplotlib import pyplot

import fix_imports

import config
from database import loadDatabase
from search import *
from filtering import Filter
from face_detector import FaceDetectorAndTracker
from scores_accumulator import ScoresAccumulator
from learning.joint_bayesian import jointBayesianDistance
from utils.file_manager import pickleLoad

from cpp_wrapper.alignment import FaceNormalization
from cpp_wrapper.descriptors import *



def getLinesFromLandmarks(shape):
    if shape.shape[0] == 51:
        shape = np.vstack((np.zeros((17, 2)), shape))
    
    lines = zip(shape[:17], shape[1:17])
    lines += zip(shape[17:22], shape[18:22]) + zip(shape[22:27], shape[23:27])
    lines += zip(shape[27:31], shape[28:31]) + zip(shape[31:36], shape[32:36])
    lines += zip(shape[36:42], shape[37:42]) + [(shape[41], shape[36])]
    lines += zip(shape[42:48], shape[43:48]) + [(shape[47], shape[42])]
    lines += zip(shape[48:60], shape[49:60]) + [(shape[48], shape[59])]
    lines += zip(shape[60:], shape[61:]) + [(shape[60], shape[67])]
    return lines


def displayShape(img, shape):
    for landmark in shape:
        cv2.circle(img, tuple(landmark.astype(np.int)), 1, (0,255,0), -1)
        
    for line in getLinesFromLandmarks(shape):
        cv2.line(img, tuple(line[0].astype(np.int)), tuple(line[1].astype(np.int)), (0,255,0))


def initDescriptor(descriptor_type, database_name, reference_shape):
    face_normalization = FaceNormalization()
    face_normalization.setReferenceShape(reference_shape)
    
    pca = Pca(filename=os.path.join(config.models_path, "PCA_%s.txt" % database_name))
    lda = Lda(os.path.join(config.models_path, "LDA_%s.txt" % database_name))
    descriptor = LbpDescriptor(descriptor_type, pca=pca, lda=lda)
    
    return face_normalization, descriptor
    

def computeDescriptor(image, (face_normalization, descriptor)):
    normalized_image = face_normalization.normalize(image, shape)
    normalized_image = normalized_image[49:201, 84:166]

    if "jb" in descriptor_type:
        jb = pickleLoad(os.path.join(config.models_path, "JB_%s.txt" % database_name))
        desc = descriptor.compute(normalized_image, normalize=False)
        return jb.transform(desc[np.newaxis]).ravel()
    else:
        return descriptor.compute(normalized_image)




if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("input_video_file", help="video to process")
    parser.add_argument("-m", dest="descriptor", default="ulbp_pca_lda", help="database")
    parser.add_argument("-d", dest="database", default="lfw_normalized_lbf_68_landmarks", help="database")
    parser.add_argument("-o", dest="output_file", help="where to write processed file")
    parser.add_argument("-n", dest="nn", type=int, default=50, help="number of neighbors in NN")
    parser.add_argument("-v", dest="verbose", action="store_true", help="verbose output")
    args = parser.parse_args()
    nn = args.nn
    descriptor_type = args.descriptor
    database_name = args.database
    verbose = args.verbose
    
    video_file = args.input_video_file
    video = cv2.VideoCapture(video_file)
    if not video.isOpened():
        raise Exception("Cannot read video %s"%video_file)

    if args.output_file:
        output_file = args.output_file
        video_writer = cv2.VideoWriter(
            output_file,
            int(video.get(6)),
            video.get(5),
            (int(video.get(3)), int(video.get(4)))
        )
        if not video_writer.isOpened():
            raise Exception("Cannot write to file %s"%output_file)
        write_output = True
    else:
        write_output = False
    
    
    detector_and_tracker = FaceDetectorAndTracker()
    descriptor = initDescriptor(descriptor_type, database_name, detector_and_tracker.alignment_with_face_detector.getReferenceShape())
    database = loadDatabase(desc=descriptor_type, db=database_name)
    filters = []

    if "jb" in descriptor_type:
        similarity = jointBayesianDistance
    else:
        similarity = np.inner

    cv2.namedWindow("Alignment demo")


    fps = video.get(5)
    print "Video running at %0.2f fps"%fps

    n = 0
    face_detector_freq = int(fps / 2)
    shapes = []
    scores_accumulator = ScoresAccumulator(max_size=int(1e4))
    
    while True:
        video.grab()
        isVideoStillReading, frame = video.retrieve()
        
        if not isVideoStillReading:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if n % face_detector_freq == 0:
            shapes = detector_and_tracker.detect(image)
            filters = []
            for i, shape in enumerate(shapes):
                filters.append(Filter(n=3))
                shapes[i] = filters[-1].filter(shape)

        elif len(shapes) > 0:
            shapes = detector_and_tracker.track(image, shapes)
            for i, shape in enumerate(shapes):
                shapes[i] = filters[i].filter(shape)
                
        else:
            filters = []

        for i, shape in enumerate(shapes):
            displayShape(frame, shape)

            desc = computeDescriptor(image, descriptor)
            nn_scores, _ = nnSumSearch(desc, database, nn, similarity=similarity)
            scores_accumulator.addScores(nn_scores[:5], n)

            if verbose:
                output = ""
                for label, score in nn_scores[:5]:
                    output += "%s: %0.2f \t"%(label, score)
                print output

        
        cv2.imshow("Alignment demo", frame)
        if cv2.waitKey(1) >= 0:
            break

        n += 1

        if write_output:
            video_writer.write(frame)


    best_labels = scores_accumulator.getBestLabels(tmax=n)
    curves = scores_accumulator.getLabelsScores(best_labels, tmax=n)
    xaxis = np.arange(n, dtype=np.float) / fps

    print "Persons detected: " + " ".join(best_labels)

    for label, curve in zip(best_labels, curves):
        pyplot.plot(xaxis, curve, label=label)
    pyplot.legend()
    pyplot.show()
