from libcpp.vector cimport vector
from cpython.ref cimport PyObject

cimport numpy as np



cdef extern from "opencv2/core/core.hpp":
    cdef cppclass _Mat "Mat":
        int rows, cols
        unsigned char* data
        int type()
        _Mat row(int)
        _Mat reshape(int, int)

    cdef cppclass _Rect "Rect":
        int x, y, width, height


cdef extern from "<iostream>" namespace "std":
    cdef cppclass ostream:
        ostream &operator << (_Mat)
    ostream cout


cdef extern from "cv2.cpp":
    void import_array()

    PyObject* pyopencv_from(const _Mat&)
    int pyopencv_to(PyObject*, _Mat&)


cdef void initMatConversion()

cdef void createCMat(np.ndarray array, _Mat& mat)
#cdef void createCMatFromDoubleArray(double[:,:] array, _Mat& mat)
cdef object createPyMat(const _Mat& mat)

cdef convertArrayToVector(np.ndarray array, vector[_Mat]& vec)

cdef void createCRect(tuple rect, _Rect& _rect)
cdef tuple createPyRect(const _Rect& _rect)
