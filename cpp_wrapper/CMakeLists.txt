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

IF(CMAKE_BUILD_TYPE MATCHES DEBUG)
    SET(debugFlag "--debug")
    MESSAGE("CMAKE_BUILD_TYPE = ${CMAKE_BUILD_TYPE}")
ENDIF(CMAKE_BUILD_TYPE MATCHES DEBUG)

add_custom_target(cython ALL
COMMAND python2 ${CMAKE_CURRENT_SOURCE_DIR}/setup.py ${debugFlag} build_ext --inplace --eigen_includes="${EIGEN_INCLUDE_DIRS}" --opencv_includes="${OpenCV_INCLUDE_DIRS}" --opencv_libraries="${OpenCV_LIB_DIR}"
COMMAND cp *.so ../lib/cpp_wrapper
)

add_custom_target(complete
COMMAND python2 setup.py ${debugFlag} build_ext --inplace --use_csiro_alignment --build_alignment_training --eigen_includes="${EIGEN_INCLUDE_DIRS}" --opencv_includes="${OpenCV_INCLUDE_DIRS}" --opencv_libraries="${OpenCV_LIB_DIR}" --liblinear_directory="${LIBLINEAR_DIR}"
COMMAND cp *.so ../lib/cpp_wrapper)


add_custom_target(cleanall
COMMAND rm -f ../lib/cpp_wrapper/*.so
COMMAND rm -rf *.so
COMMAND rm -f *.cpp
COMMAND rm -f *.c
COMMAND rm -rf build
COMMAND rm -rf */*.o
)
  
  
#add_custom_target(debug)

#add_custom_target(clean)
