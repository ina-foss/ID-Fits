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


cmake_minimum_required (VERSION 2.8 FATAL_ERROR)

############################################################################################################################
Project(ID-Fits)
set (${CMAKE_PROJECT_NAME}_VERSION_MAJOR 1)
set (${CMAKE_PROJECT_NAME}_VERSION_MINOR 0)
set (${CMAKE_PROJECT_NAME}_VERSION_PATCHLEVEL 0)
set (PROJECT_VERSION ${${CMAKE_PROJECT_NAME}_VERSION_MAJOR}.${${CMAKE_PROJECT_NAME}_VERSION_MINOR}.${${CMAKE_PROJECT_NAME}_VERSION_PATCHLEVEL})
set (SYNOPSIS "Synopsis of the ${CMAKE_PROJECT_NAME} project")


############################################################################################################################

#SET(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_COLOR_MAKEFILE ON)
set(SO_VERSION ${PROJECT_VERSION})
set(VERSION 0)

############################################################################################################################


IF ( NOT DEFINED EIGEN_INCLUDE_DIRS)
    MESSAGE(FATAL_ERROR "You need to specify EIGEN_INCLUDE_DIRS: an Eigen3 include directory")
    
ELSEIF(NOT EXISTS "${EIGEN_INCLUDE_DIRS}/eigen3/Eigen/Eigen")
        MESSAGE(FATAL_ERROR "\"${EIGEN_INCLUDE_DIRS}\" is not a valid Eigen3 include directory")
ENDIF()    


find_package(OpenCV REQUIRED)

IF ( NOT DEFINED OpenCV_INCLUDE_DIRS)
    MESSAGE(FATAL_ERROR "You need to specify OpenCV_DIR: an OpenCV-install-dir/share/OpenCV directory")
ENDIF()

IF ( NOT DEFINED LIBLINEAR_DIR)
    MESSAGE(FATAL_ERROR "You need to specify LIBLINEAR_DIR: a LibLinear directory")
    
ELSEIF(NOT EXISTS "${LIBLINEAR_DIR}/liblinear/linear.h")
        MESSAGE(FATAL_ERROR "\"${LIBLINEAR_DIR}\" is not a valid LibLinear directory")
ENDIF() 


############################################################################################################################


add_subdirectory(cpp_wrapper)


add_custom_target(distclean
    COMMAND rm -rf CMakeFiles CMakeCache.txt cmake_install.cmake Makefile
    COMMAND rm -rf cpp_wrapper/CMakeFiles cpp_wrapper/cmake_install.cmake cpp_wrapper/Makefile cpp_wrapper/cython_debug
)

add_dependencies(distclean cleanall)
