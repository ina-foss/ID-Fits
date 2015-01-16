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


from cpython.ref cimport PyObject

import numpy as np
cimport numpy as np

from opencv_types cimport *



cdef void initMatConversion():
    import_array()


cdef void createCMat(np.ndarray array, _Mat& mat):
    cdef PyObject* pyobject = <PyObject*> array
    pyopencv_to(pyobject, mat)


"""
cdef void createCMatFromDoubleArray(double[:,:] array, _Mat& mat):
    cdef PyObject* pyobject = <PyObject*> array
    pyopencv_to(pyobject, mat)
"""


cdef object createPyMat(const _Mat& mat):
    return <object> pyopencv_from(mat)


cdef convertArrayToVector(np.ndarray array, vector[_Mat]& vec):
    cdef _Mat temp

    for i in range(len(array)):
        vec.push_back(temp)
        createCMat(array[i], vec[i])


cdef void createCRect(tuple rect, _Rect& _rect):
    _rect.x = rect[0]
    _rect.y = rect[1]
    _rect.width = rect[2] - rect[0]
    _rect.height = rect[3] - rect[1]


cdef tuple createPyRect(const _Rect& _rect):
    return (_rect.x, _rect.y, _rect.x+_rect.width, _rect.y+_rect.height)
