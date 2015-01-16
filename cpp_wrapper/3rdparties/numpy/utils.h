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


#ifndef __NUMPY_UTILS_HEADER__
#define __NUMPY_UTILS_HEADER__

#ifndef __COMP_NPY_UNUSED
        #if defined(__GNUC__)
                #define __COMP_NPY_UNUSED __attribute__ ((__unused__))
        # elif defined(__ICC)
                #define __COMP_NPY_UNUSED __attribute__ ((__unused__))
        #else
                #define __COMP_NPY_UNUSED
        #endif
#endif

/* Use this to tag a variable as not used. It will remove unused variable
 * warning on support platforms (see __COM_NPY_UNUSED) and mangle the variable
 * to avoid accidental use */
#define NPY_UNUSED(x) (__NPY_UNUSED_TAGGED ## x) __COMP_NPY_UNUSED

#endif
