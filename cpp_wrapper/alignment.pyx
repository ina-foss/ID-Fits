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


from libcpp.vector cimport vector
from libcpp.string cimport string
from cython.operator cimport dereference as deref

import os
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


    
if "USE_CSIRO_ALIGNMENT" in os.environ and os.environ["USE_CSIRO_ALIGNMENT"] == 1:
    DEF USE_CSIRO_ALIGNMENT = 1
else:
    DEF USE_CSIRO_ALIGNMENT = 0

    
IF USE_CSIRO_ALIGNMENT:

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

        if landmarks not in [51, 68]:
            raise Exception("Wrong landmarks number")
        else:
            landmarks_str = "%i_landmarks" % landmarks

        if detector == "opencv":
            detector_str = "opencv_detector"
        elif detector == "estimated":
            detector_str = "estimated_bounding_box"
        elif detector == "perfect":
            detector_str = "perfect_detector"
        else:
            raise Exception("Wrong detector argument")

        model_file = os.path.join(config.models_path, "alignment", "lbf_regression_model_%s_%s.bin" % (landmarks_str, detector_str))

        self.thisptr = new _LBFLandmarkDetector()
        (<_LBFLandmarkDetector*> self.thisptr).loadModel(model_file)


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




