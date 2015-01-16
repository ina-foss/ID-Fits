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


#ifndef _NPY_OS_H_
#define _NPY_OS_H_

#if defined(linux) || defined(__linux) || defined(__linux__)
    #define NPY_OS_LINUX
#elif defined(__FreeBSD__) || defined(__NetBSD__) || \
            defined(__OpenBSD__) || defined(__DragonFly__)
    #define NPY_OS_BSD
    #ifdef __FreeBSD__
        #define NPY_OS_FREEBSD
    #elif defined(__NetBSD__)
        #define NPY_OS_NETBSD
    #elif defined(__OpenBSD__)
        #define NPY_OS_OPENBSD
    #elif defined(__DragonFly__)
        #define NPY_OS_DRAGONFLY
    #endif
#elif defined(sun) || defined(__sun)
    #define NPY_OS_SOLARIS
#elif defined(__CYGWIN__)
    #define NPY_OS_CYGWIN
#elif defined(_WIN32) || defined(__WIN32__) || defined(WIN32)
    #define NPY_OS_WIN32
#elif defined(__APPLE__)
    #define NPY_OS_DARWIN
#else
    #define NPY_OS_UNKNOWN
#endif

#endif
