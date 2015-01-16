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


#ifndef _NPY_NUMPYCONFIG_H_
#define _NPY_NUMPYCONFIG_H_

#include "_numpyconfig.h"

/*
 * On Mac OS X, because there is only one configuration stage for all the archs
 * in universal builds, any macro which depends on the arch needs to be
 * harcoded
 */
#ifdef __APPLE__
    #undef NPY_SIZEOF_LONG
    #undef NPY_SIZEOF_PY_INTPTR_T

    #ifdef __LP64__
        #define NPY_SIZEOF_LONG         8
        #define NPY_SIZEOF_PY_INTPTR_T  8
    #else
        #define NPY_SIZEOF_LONG         4
        #define NPY_SIZEOF_PY_INTPTR_T  4
    #endif
#endif

/**
 * To help with the NPY_NO_DEPRECATED_API macro, we include API version
 * numbers for specific versions of NumPy. To exclude all API that was
 * deprecated as of 1.7, add the following before #including any NumPy
 * headers:
 *   #define NPY_NO_DEPRECATED_API  NPY_1_7_API_VERSION
 */
#define NPY_1_7_API_VERSION 0x00000007
#define NPY_1_8_API_VERSION 0x00000008

#endif
