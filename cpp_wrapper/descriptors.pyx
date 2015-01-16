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


from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.string cimport string
from cython.operator import dereference as deref

import numpy as np
cimport numpy as np
import cPickle as pickle

from opencv_types cimport *
from descriptors cimport *


cdef extern from "Pca.h":
    cdef cppclass _Pca "Pca":
        _Pca()
        _Pca(const _Mat&, const _Mat&, const _Mat&)
        void create(const vector[_Mat]&, int dim)
        void project(const _Mat&, _Mat&) const
        void save(const string&) const
        void load(const string&)
        _Mat getEigenvalues() const
        void reduceDimension(int dim)


cdef extern from "Lda.h":
    cdef cppclass _Lda "Lda":
        _Lda(const _Mat&, const _Mat&)
        void project(const _Mat&, _Mat&) const


cdef extern from "Descriptors.h":
    cdef cppclass _Descriptor "Descriptor":
        _Descriptor()
        _Descriptor(int, int)
        void compute(const _Mat&, _Mat&, bool normalize) const
    
    cdef cppclass _LbpDescriptor "LbpDescriptor" (_Descriptor):
        _LbpDescriptor()
        _LbpDescriptor(int, int)
    
    cdef cppclass _ULbpDescriptor "ULbpDescriptor" (_LbpDescriptor):
        _ULbpDescriptor()
        _ULbpDescriptor(int, int)
    
    cdef cppclass _ULbpPCADescriptor "ULbpPCADescriptor" (_ULbpDescriptor):
        # FIXME why is it needed ???
        _ULbpPCADescriptor()
        _ULbpPCADescriptor(_Pca)
        _ULbpPCADescriptor(_Pca, _Lda)
    
    cdef cppclass _ULbpWPCADescriptor "ULbpWPCADescriptor" (_ULbpPCADescriptor):
        _ULbpWPCADescriptor(_Pca)

    cdef cppclass _ULbpPCALDADescriptor "ULbpPCALDADescriptor" (_ULbpPCADescriptor):
        _ULbpPCALDADescriptor(_Pca, _Lda)

    cdef cppclass _PCAReducedDescriptor "PCAReducedDescriptor":
        _PCAReducedDescriptor(const _Descriptor*, const _Pca&)
        void compute(const _Mat&, _Mat&) const


cdef class Pca:
    cdef _Pca *thisptr

    def __cinit__(self, mean = None, eigenvalues = None, eigenvectors = None, filename = None, sklearn_pca = None):
        cdef _Mat _mean, _eigenvalues, _eigenvectors
        initMatConversion()
        
        if mean is not None or sklearn_pca is not None:
            if sklearn_pca is not None:
                mean = sklearn_pca.mean_[np.newaxis, :]
                eigenvalues = sklearn_pca.explained_variance_[:, np.newaxis]
                eigenvectors = sklearn_pca.components_

            createCMat(mean, _mean)
            createCMat(eigenvalues, _eigenvalues)
            createCMat(eigenvectors, _eigenvectors)
            self.thisptr = new _Pca(_mean, _eigenvalues, _eigenvectors)
        else:
            self.thisptr = new _Pca()
            if filename is not None and type(filename) is str:
                self.load(filename)
            

    def __dealloc__(self):
        del self.thisptr

    def create(self, learning_data, dim):
        cdef vector[_Mat] _learning_data
        convertArrayToVector(learning_data, _learning_data)
        self.thisptr.create(_learning_data, dim)
    
    def project(self, src):
        cdef _Mat _src, _dst
        createCMat(src, _src)
        self.thisptr.project(_src, _dst)
        return createPyMat(_dst).ravel()

    def save(self, string filename):
        self.thisptr.save(filename)

    def load(self, string filename):
        self.thisptr.load(filename)

    def getEigenvalues(self):
        return createPyMat(self.thisptr.getEigenvalues()).ravel()

    def reduceDimension(self, int dim):
        self.thisptr.reduceDimension(dim)

    property eigenvalues:
        def __get__(self): return self.getEigenvalues()


cdef class Lda:
    cdef _Lda *thisptr

    def __cinit__(self, filename):
        cdef _Mat _mean, _scalings
        with open(filename, "r") as f:
            lda = pickle.load(f)
        initMatConversion()
        createCMat(lda.xbar_.astype(np.float32), _mean)
        createCMat(lda.scalings_[:, :lda.n_components].astype(np.float32), _scalings)
        self.thisptr = new _Lda(_mean, _scalings)

    def __dealloc__(self):
        del self.thisptr

    def project(self, src):
        cdef _Mat _src, _dst
        createCMat(src, _src)
        self.thisptr.project(_src, _dst)
        return createPyMat(_dst).ravel() 



cdef class LbpDescriptor:
    cdef _Descriptor *thisptr

    def __cinit__(self, descriptor, cell_size=10, step=-1, Pca pca=None, Lda lda=None, pca_on_each_hist=False):
        initMatConversion()

        if descriptor == "lbp":
            self.thisptr = new _LbpDescriptor(cell_size, step)
        elif descriptor == "ulbp":
            self.thisptr = new _ULbpDescriptor(cell_size, step)
        elif descriptor == "ulbp_pca":
            self.thisptr = new _ULbpPCADescriptor(deref(pca.thisptr))
        elif descriptor == "ulbp_wpca":
            self.thisptr = new _ULbpWPCADescriptor(deref(pca.thisptr))
        elif descriptor == "ulbp_pca_lda":
            self.thisptr = new _ULbpPCALDADescriptor(deref(pca.thisptr), deref(lda.thisptr))
        elif descriptor == "ulbp_pca_jb":
            self.thisptr = new _ULbpPCADescriptor(deref(pca.thisptr))
        else:
            raise Exception("Unknown descriptor")

        """
        if pca_on_each_hist:
            self.descriptor = self.thisptr
            self.thisptr = new _PCAReducedDescriptor(self.descriptor, deref(pca.thisptr))
        else:
            self.descriptor = None
        """

        
    def __dealloc__(self):
        del self.thisptr
        """
        if self.descriptor:
            del self.descriptor
        """


    def compute(self, np.ndarray[char, ndim=2]  img, bool normalize = True, flatten = True):
        cdef _Mat _img, _desc
        createCMat(img, _img)
        self.thisptr.compute(_img, _desc, normalize)
        if flatten:
            return createPyMat(_desc).ravel()
        else:
            return createPyMat(_desc)



cdef class Hog:
    def __cinit__(self):
        self.thisptr = new _Hog()
        initMatConversion()


    def __dealloc__(self):
        del self.thisptr


    cpdef init(self):
        initMatConversion()

    cpdef double[:,:] compute(self, char[:,:] src):
        cdef _Mat mat
        createCMat(np.asarray(src), mat)
        return createPyMat(self.thisptr.compute(mat))


    cpdef np.ndarray[double, ndim=1] compute_(self, np.ndarray[char, ndim=2] src):
        cdef _Mat mat
        createCMat(src, mat)
        return createPyMat(self.thisptr.compute(mat))

