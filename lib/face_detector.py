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

from cpp_wrapper.face_detection import FaceDetector
from cpp_wrapper.alignment import LBFLandmarkDetector



class FaceDetectorAndTracker:
    
    def __init__(self):
        self.detector = FaceDetector()
        self.alignment_with_face_detector = LBFLandmarkDetector(detector="opencv", landmarks=68)
        self.alignment_for_tracking = LBFLandmarkDetector(detector="perfect", landmarks=68)
    
    
    def detect(self, image):
        faces = self.detector.detectFaces(image)
        shapes = []
        
        for face in faces:
            shapes.append(self.alignment_with_face_detector.detectLandmarks(image, face))
        
        return shapes
    
    
    def track(self, image, shapes):
        new_shapes = []
        
        for shape in shapes:
            old_shape = shape.astype(np.int)
            face = (np.min(old_shape[:,0]), np.min(old_shape[:,1]), np.max(old_shape[:,0]), np.max(old_shape[:,1]))
            new_shapes.append(self.alignment_for_tracking.detectLandmarks(image, face))
        
        return new_shapes

