import numpy as np

from cpp_wrapper.face_detection import FaceDetector
from cpp_wrapper.alignment import LBFLandmarkDetector



class FaceDetectorAndTracker:
    
    def __init__(self):
        self.detector = FaceDetector()
        self.alignment_with_face_detector = LBFLandmarkDetector(detector="opencv", landmarks=68)
        self.alignment_for_tracking = LBFLandmarkDetector(detector="estimated", landmarks=68)
    
    
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

