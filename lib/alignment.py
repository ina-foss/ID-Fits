import time
import cv2

from tools import *
from cpp_wrapper.face_detection import *
from cpp_wrapper.alignment import *


class lfwFaceDetector:
    def __init__(self):
        self.face_detector = FaceDetector()
        self.high_recall_face_detector = FaceDetector(high_recall=True)

    def detect(self, image):
        not_detected_bounding_box = (0, 0, image.shape[0]-1, image.shape[1]-1)
        bounding_boxes = self.face_detector.detectFaces(image)
        if len(bounding_boxes) > 0:
            return bounding_boxes[0]
        else:
            bounding_boxes = self.high_recall_face_detector.detectFaces(image)
            if len(bounding_boxes) > 0:
                return bounding_boxes[0]
            else:
                return not_detected_bounding_box


def normalizeData(data, landmark_detector_name="lbf", landmarks_number=None):

    if landmark_detector_name == "lbf":
        face_detector = FaceDetector()
        high_recall_face_detector = FaceDetector(high_recall=True)
        if landmarks_number is None:
            landmark_detector = LBFLandmarkDetector(detector="opencv")
        else:
            landmark_detector = LBFLandmarkDetector(detector="opencv", landmarks=landmarks_number)
    elif landmark_detector_name == "csiro":
         landmark_detector = CSIROLandmarkDetector()
    else:
        raise Exception("Unknown landmark detector: %s"%landmark_detector)
        
    face_normalization = FaceNormalization()
    face_normalization.setReferenceShape(landmark_detector.getReferenceShape())
    
    undetected_faces = 0
    low_precision_faces = 0
    t0 = time.clock()
    t = t0
    normalized_data = []
    for i,img in enumerate(data):
        if landmark_detector_name == "lbf":
            bounding_boxes = high_recall_face_detector.detectFaces(img)
            if len(bounding_boxes) == 0:
                undetected_faces += 1
                bounding_box = (0, 0, img.shape[0]-1, img.shape[1]-1)
            else:
                bounding_box = bounding_boxes[0]
                bounding_boxes = face_detector.detectFaces(img)
                if len(bounding_boxes) > 0:
                    bounding_box = bounding_boxes[0]
                else:
                    low_precision_faces += 1
            
            landmarks = landmark_detector.detectLandmarks(img, bounding_box)
        else:
            landmarks = landmark_detector.detectLandmarks(img)

        landmarks = landmark_detector.extractLandmarksForNormalization(landmarks)
        normalized_data.append(face_normalization.normalize(img, landmarks))
        if time.clock() - t > 10.0:
            print "%.02f%% of images normalized"%(100*i/float(len(data)))
            t = time.clock()

    if undetected_faces > 0:
        print "%d faces not detected"%undetected_faces
        print "%d faces detected only by high recall detector"%low_precision_faces
    print "%d total number of images"%len(data)

    return np.asarray(normalized_data)



def markLandmarks(imgs, color=(255,0,0)):
    landmark_detector = CSIROLandmarkDetector()
    for img in imgs:
        landmarks = landmark_detector.detectLandmarks(img)
        for landmark in landmarks:
            cv2.circle(img, tuple(landmark.astype(np.int)), 2, color, -1)
