cimport numpy as np
from opencv_types cimport *


cdef extern from "Hog.h":
    cdef cppclass _Hog "Hog":
        _Mat compute(const _Mat&)

cdef class Hog:
    cdef _Hog* thisptr

    cpdef init(self)
    cpdef double[:,:] compute(self, char[:,:] src)
    cpdef np.ndarray[double, ndim=1] compute_(self, np.ndarray[char, ndim=2] src)

