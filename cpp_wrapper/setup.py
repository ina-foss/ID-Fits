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


import sys
import os
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize


macros = []

desc_dir = "../cpp/SnoopFaceDescLib/"
alignment_dir = "../cpp/SnoopFaceAlignmentLib/"
include_dirs = [
    "/usr/include/eigen3",
    "/home/tlorieul/Dev/install/include/",
    "../cpp/",
    "../cpp/SnoopFaceDescLib",
    "../cpp/SnoopFaceAlignmentLib",
    "3rdparties/"]
library_dirs = ["/home/tlorieul/Dev/install/lib"]
opencv_libs = ["opencv_core", "opencv_highgui", "opencv_imgproc", "opencv_objdetect"]
cxx_flags = ["-Wno-unused-function", "-msse2", "-DNDEBUG"]


extensions = [
    Extension("opencv_types", ["opencv_types.pyx"],
              language = "c++",
              extra_compile_args = cxx_flags,
              include_dirs = include_dirs,
              libraries = opencv_libs,
              library_dirs = library_dirs),
    Extension("descriptors", ["descriptors.pyx", alignment_dir + "Hog.cc"],
              language = "c++",
              extra_compile_args = cxx_flags,
              include_dirs = include_dirs,
              libraries = opencv_libs,
              library_dirs = library_dirs),
    Extension("alignment", ["alignment.pyx", alignment_dir + "ForestBasedRegression.cc", alignment_dir + "Alignment.cc"],
              language = "c++",
              define_macros = [("USE_CSIRO_ALIGNMENT", 0)],
              extra_compile_args = cxx_flags,
              include_dirs = include_dirs,
              libraries = opencv_libs,
              library_dirs = library_dirs),
    Extension("face_detection", ["face_detection.pyx"],
              language = "c++",
              extra_compile_args = cxx_flags,
              include_dirs = include_dirs,
              libraries = opencv_libs,
              library_dirs = library_dirs)
]


if "use_csiro_alignment" in sys.argv:
    sys.argv.remove("use_csiro_alignment")
    os.environ["USE_CSIRO_ALIGNMENT"] = "1"
    extensions[2] = Extension("alignment", ["alignment.pyx", alignment_dir + "ForestBasedRegression.cc", alignment_dir + "Alignment.cc"],
                              language = "c++",
                              define_macros = [("USE_CSIRO_ALIGNMENT", 1)],
                              extra_compile_args = cxx_flags,
                              include_dirs = include_dirs,
                              libraries = opencv_libs + ["clmTracker"],
                              library_dirs = library_dirs)


if "build_alignment_training" in sys.argv:
    sys.argv.remove("build_alignment_training")
    extensions.append(Extension("tree_based_regression", ["tree_based_regression.pyx", alignment_dir + "ForestBasedRegression.cc"],
                                language = "c++",
                                extra_compile_args = cxx_flags + ["-fopenmp", "-ffast-math", "-O3"],
                                extra_link_args = ["-fopenmp", "-ffast-math"],
                                include_dirs = include_dirs,
                                libraries = opencv_libs + ["boost_random", "linear"],
                                library_dirs = library_dirs)
                  )
    


if "debug" in sys.argv:
    sys.argv.remove("debug")
    setup(
        name = "descriptors",
        ext_modules = cythonize(extensions, gdb_debug=True)
    )
else:
    setup(
        name = "descriptors",
        ext_modules = cythonize(extensions)
    )
