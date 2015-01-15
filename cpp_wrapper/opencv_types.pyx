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
