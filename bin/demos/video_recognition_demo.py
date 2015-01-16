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

import fix_imports

from database import loadDatabase
from search import *
from face_detector import FaceDetectorAndTracker

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


def initDescriptor(reference_shape):
    face_normalization = FaceNormalization()
    face_normalization.setReferenceShape(reference_shape)
    pca = Pca(filename="PCA/ulbp_normalized_data_lbf_alignment_68_landmarks/PCA_set_1.txt")
    lda = Lda("LDA/ulbp_normalized_data_lbf_alignment_68_landmarks/set_1.txt")
    descriptor = ULbpPCALDADescriptor(pca, lda)
    return face_normalization, descriptor, lda
    

def computeDescriptor(image, (face_normalization, descriptor, lda)):
    image = cv2.cvtColor(image, 6)
    face_normalization.normalize(image, shape)
    image = image[49:201, 84:166]
    return descriptor.computeDescriptor(image)




if __name__ == "__main__":
    """
    video_file = "/home/tlorieul/Data/ytcelebrity/0772_01_003_hillary_clinton.avi"
    video_file = "/home/tlorieul/Data/ytcelebrity/0057_01_002_al_gore.avi"
    """
    video_file = "/home/tlorieul/dwhelper/Hillary_Clinton_s_Relationship_With_Obama_7215.mp4"
    
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("input_video_file", help="video to process")
    parser.add_argument("-o", dest="output_file", help="where to write processed file")
    parser.add_argument("-n", dest="nn", type=int, default=50, help="number of neighbors in NN")
    args = parser.parse_args()
    nn = args.nn
    
    video_file = args.input_video_file
    video = cv2.VideoCapture(video_file)
    if not video.isOpened():
        raise Exception("Cannot read video %s"%video_file)

    if args.output_file:
        output_file = args.output_file
        video_writer = cv2.VideoWriter(
            output_file,
            877677894,
            video.get(5),
            (int(video.get(3)), int(video.get(4)))
        )
        if not video_writer.isOpened():
            raise Exception("Cannot write to file %s"%output_file)
        write_output = True
    else:
        write_output = False
    
    
    detector_and_tracker = FaceDetectorAndTracker()
    descriptor = initDescriptor(detector_and_tracker.alignment_with_face_detector.getReferenceShape())
    database = loadDatabase(desc="ulbp_pca_lda", db="normalized_data_lbf_alignment_68_landmarks")
    
    cv2.namedWindow("Alignment demo")


    fps = video.get(5)
    print "Video running at %0.2f fps"%fps

    n = 0
    face_detector_freq = int(fps / 2)
    shapes = []
    
    while True:
        video.grab()
        isVideoStillReading, frame = video.retrieve()
        
        if not isVideoStillReading:
            break

        image = np.copy(frame)
        
        if n % face_detector_freq == 0:
            shapes = detector_and_tracker.detect(image)
        elif len(shapes) > 0:
            shapes = detector_and_tracker.track(image, shapes)

        for shape in shapes:
            displayShape(frame, shape)

            desc = computeDescriptor(image, descriptor)
            nn_scores, _ = nnSumSearch(desc, database, nn)

            output = ""
            for label, score in nn_scores[:5]:
                output += "%s: %0.2f \t\t"%(label, score)
            print output

        
        cv2.imshow("Alignment demo", frame)
        if cv2.waitKey(25) >= 0:
            break

        n += 1

        if write_output:
            video_writer.write(frame)
