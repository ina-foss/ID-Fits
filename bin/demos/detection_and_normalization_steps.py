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

from cpp_wrapper.face_detection import FaceDetector
from cpp_wrapper.alignment import LBFLandmarkDetector, FaceNormalization



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Simple program detecting faces and detecting facial landmarks.")
    parser.add_argument("image_file", help="input image")
    args = parser.parse_args()


    # Read the image
    image = cv2.imread(args.image_file, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    copy = cv2.imread(args.image_file)
    clean_copy = np.copy(copy)

    cv2.imshow("Original image", copy)
    cv2.waitKey(0)


    # Detect the faces
    detector = FaceDetector()
    faces = detector.detectFaces(image)
    if len(faces) < 1:
        raise Exception("No faces detected")

    for face in faces:
        cv2.rectangle(copy, face[:2], face[2:], (0,0,255), 3)

    cv2.imshow("Face detector", copy)
    cv2.waitKey(0)


    # Detect the landmarks
    alignment = LBFLandmarkDetector(detector="opencv", landmarks=51)
    shapes = [alignment.detectLandmarks(image, face) for face in faces]

    for shape in shapes:
        for landmark in shape:
            cv2.circle(copy, tuple(landmark.astype(np.int)), 2, (0,255,0), -1)

    cv2.imshow("Landmarks detector", copy)
    cv2.waitKey(0)


    # Normalize the face
    face_normalization = FaceNormalization()
    face_normalization.setReferenceShape(alignment.getReferenceShape())

    for i, shape in enumerate(shapes):
        normalized_image = face_normalization.normalize(np.copy(copy), shape)
        cv2.imshow("Normalized face %i / %i"%(i+1, len(shapes)), normalized_image)
        cv2.waitKey(0)

        normalized_image = face_normalization.normalize(np.copy(clean_copy), shape)
        cv2.imshow("Normalized face %i / %i"%(i+1, len(shapes)), normalized_image)
        cv2.waitKey(0)
