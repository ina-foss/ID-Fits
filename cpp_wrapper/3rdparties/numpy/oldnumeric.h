// ID-Fits
// Copyright (c) 2015 Institut National de l'Audiovisuel, INA, All rights reserved.
// 
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 3.0 of the License, or (at your option) any later version.
// 
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
// 
// You should have received a copy of the GNU Lesser General Public
// License along with this library.


#include "arrayobject.h"

#ifndef REFCOUNT
#  define REFCOUNT NPY_REFCOUNT
#  define MAX_ELSIZE 16
#endif

#define PyArray_UNSIGNED_TYPES
#define PyArray_SBYTE NPY_BYTE
#define PyArray_CopyArray PyArray_CopyInto
#define _PyArray_multiply_list PyArray_MultiplyIntList
#define PyArray_ISSPACESAVER(m) NPY_FALSE
#define PyScalarArray_Check PyArray_CheckScalar

#define CONTIGUOUS NPY_CONTIGUOUS
#define OWN_DIMENSIONS 0
#define OWN_STRIDES 0
#define OWN_DATA NPY_OWNDATA
#define SAVESPACE 0
#define SAVESPACEBIT 0

#undef import_array
#define import_array() { if (_import_array() < 0) {PyErr_Print(); PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import"); } }
