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


from libcpp.string cimport string
from opencv_types cimport *
import config


cdef extern from "FaceDetection.h":

    cdef cppclass _FaceDetector "FaceDetector":
        _FaceDetector()
        _FaceDetector(const string&)
        void detectFaces(const _Mat&, vector[_Rect]&)


    cdef cppclass _HighRecallFaceDetector "HighRecallFaceDetector" (_FaceDetector):
        _HighRecallFaceDetector(const string&)



cdef class FaceDetector:
    cdef _FaceDetector *thisptr


    def __cinit__(self, high_recall = False):
        initMatConversion()
        model_file = config.models_path + "/detection/haarcascade_frontalface_alt.xml"
        if not high_recall:
            self.thisptr = new _FaceDetector(model_file)
        else:
            self.thisptr = new _HighRecallFaceDetector(model_file)


    def __dealloc__(self):
        del self.thisptr


    def detectFaces(self, img):
        cdef:
            _Mat _img
            vector[_Rect] _faces
        createCMat(img, _img)
        self.thisptr.detectFaces(_img, _faces)
        return [createPyRect(_faces[i]) for i in range(_faces.size())]
