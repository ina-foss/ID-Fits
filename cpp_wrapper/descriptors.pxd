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

