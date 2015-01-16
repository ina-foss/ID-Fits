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


#ifndef _NPY_ENDIAN_H_
#define _NPY_ENDIAN_H_

/*
 * NPY_BYTE_ORDER is set to the same value as BYTE_ORDER set by glibc in
 * endian.h
 */

#ifdef NPY_HAVE_ENDIAN_H
    /* Use endian.h if available */
    #include <endian.h>

    #define NPY_BYTE_ORDER __BYTE_ORDER
    #define NPY_LITTLE_ENDIAN __LITTLE_ENDIAN
    #define NPY_BIG_ENDIAN __BIG_ENDIAN
#else
    /* Set endianness info using target CPU */
    #include "npy_cpu.h"

    #define NPY_LITTLE_ENDIAN 1234
    #define NPY_BIG_ENDIAN 4321

    #if defined(NPY_CPU_X86)            \
            || defined(NPY_CPU_AMD64)   \
            || defined(NPY_CPU_IA64)    \
            || defined(NPY_CPU_ALPHA)   \
            || defined(NPY_CPU_ARMEL)   \
            || defined(NPY_CPU_AARCH64) \
            || defined(NPY_CPU_SH_LE)   \
            || defined(NPY_CPU_MIPSEL)
        #define NPY_BYTE_ORDER NPY_LITTLE_ENDIAN
    #elif defined(NPY_CPU_PPC)          \
            || defined(NPY_CPU_SPARC)   \
            || defined(NPY_CPU_S390)    \
            || defined(NPY_CPU_HPPA)    \
            || defined(NPY_CPU_PPC64)   \
            || defined(NPY_CPU_ARMEB)   \
            || defined(NPY_CPU_SH_BE)   \
            || defined(NPY_CPU_MIPSEB)  \
            || defined(NPY_CPU_M68K)
        #define NPY_BYTE_ORDER NPY_BIG_ENDIAN
    #else
        #error Unknown CPU: can not set endianness
    #endif
#endif

#endif
