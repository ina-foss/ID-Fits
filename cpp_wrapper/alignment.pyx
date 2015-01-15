from libcpp.vector cimport vector
from libcpp.string cimport string
from cython.operator cimport dereference as deref

import numpy as np
cimport numpy as np

import config
from opencv_types cimport *



cdef extern from "Alignment.h":

    cdef cppclass _LandmarkDetector "LandmarkDetector":
        void detectLandmarks(const _Mat&, const _Rect&, _Mat&) const
        void extractLandmarksForNormalization(const _Mat&, _Mat&) const
        const _Mat& getReferenceShape()

    cdef cppclass _CSIROLandmarkDetector "CSIROLandmarkDetector" (_LandmarkDetector):
        _CSIROLandmarkDetector(const string&, const string&)

    cdef cppclass _LBFLandmarkDetector "LBFLandmarkDetector" (_LandmarkDetector):
        void loadModel(const string&)

    cdef cppclass _FaceNormalization "FaceNormalization":
        void setReferenceShape(const _Mat&)
        void normalize(_Mat&, _Mat&) const


cdef class LandmarkDetector:
    cdef _LandmarkDetector *thisptr

    def extractLandmarksForNormalization(self, landmarks):
        cdef _Mat _landmarks, _normalization_landmarks
        createCMat(landmarks, _landmarks)
        self.thisptr.extractLandmarksForNormalization(_landmarks, _normalization_landmarks)
        return createPyMat(_normalization_landmarks)

    def getReferenceShape(self):
        return createPyMat(self.thisptr.getReferenceShape())


cdef class CSIROLandmarkDetector(LandmarkDetector):

    def __cinit__(self):
        initMatConversion()
        self.thisptr = new _CSIROLandmarkDetector(config.models_path + "/csiro_alignment/face.mytracker", config.models_path + "/csiro_alignment/face.mytrackerparams")


    def __dealloc__(self):
        del self.thisptr
    

    def detectLandmarks(self, np.ndarray img):
        cdef:
            _Mat _img, _landmarks
            _Rect _empty
        createCMat(img, _img)
        self.thisptr.detectLandmarks(_img, _empty, _landmarks)
        return createPyMat(_landmarks)
 


cdef class LBFLandmarkDetector(LandmarkDetector):

    cdef _LBFLandmarkDetector* ptr

    def __cinit__(self, detector="opencv", landmarks=51):
        initMatConversion()
        
        if landmarks != 51 and landmarks != 68:
            raise Exception("Wrong landmarks number")

        if detector == "opencv":
            if landmarks == 51:
                model_file = "alignment/lbf_regression_model_51_landmarks_opencv_detector.txt"
            else:
                model_file = "alignment/lbf_regression_model_68_landmarks_opencv_detector.txt"
        elif detector == "estimated":
            if landmarks == 51:
                model_file = "alignment/lbf_regression_model_51_landmarks_estimated_bounding_box.txt"
            else:
                model_file = "alignment/lbf_regression_model_68_landmarks_estimated_bounding_box.txt"
        else:
            raise Exception("Wrong detector argument")

        self.thisptr = new _LBFLandmarkDetector()
        (<_LBFLandmarkDetector*> self.thisptr).loadModel(config.models_path + "/" + model_file)


    def __dealloc__(self):
        del self.thisptr


    def detectLandmarks(self, img, bounding_box):
        cdef:
            _Mat _img, _landmarks
            _Rect _bounding_box
        createCMat(img, _img)
        createCRect(bounding_box, _bounding_box)
        self.thisptr.detectLandmarks(_img, _bounding_box, _landmarks)
        return createPyMat(_landmarks)



cdef class FaceNormalization:

    cdef _FaceNormalization *thisptr


    def __cinit__(self):
        initMatConversion()
        self.thisptr = new _FaceNormalization()


    def __dealloc__(self):
        del self.thisptr


    def setReferenceShape(self, reference_landmarks):
        cdef _Mat _reference_landmarks
        createCMat(reference_landmarks, _reference_landmarks)
        self.thisptr.setReferenceShape(_reference_landmarks)


    def normalize(self, np.ndarray img, landmarks):
        cdef _Mat _img, _landmarks
        createCMat(img, _img)
        createCMat(landmarks, _landmarks)
        self.thisptr.normalize(_img, _landmarks)
        return createPyMat(_img)

