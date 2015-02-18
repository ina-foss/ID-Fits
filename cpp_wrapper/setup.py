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
import argparse



macros = []

desc_dir = "../cpp/SnoopFaceDescLib/"
alignment_dir = "../cpp/SnoopFaceAlignmentLib/"


parser = argparse.ArgumentParser(description="Setup script")
parser.add_argument("--debug", action='store_true')
parser.add_argument("--build_alignment_training", action='store_true')
parser.add_argument("--use_csiro_alignment", action='store_true')
parser.add_argument("--eigen_includes", nargs="?", help="Eigen includes directory")
parser.add_argument("--opencv_includes", nargs="?", help="OpenCV includes directory")
parser.add_argument("--opencv_libraries", nargs="?", help="OpenCV libraries directory")
parser.add_argument("--liblinear_directory", nargs="?", help="LibLinear directory")
args, unknown_args = parser.parse_known_args()
unknown_args.insert(0, 'setup.py')
sys.argv = unknown_args

include_dirs = [
    "../cpp/",
    "../cpp/SnoopFaceDescLib",
    "../cpp/SnoopFaceAlignmentLib",
    "3rdparties/"]


if args.eigen_includes:
    include_dirs.append(args.eigen_includes) 

if args.opencv_includes:
    include_dirs.extend(args.opencv_includes.split()) 
    
if args.liblinear_directory:
    include_dirs.extend(args.liblinear_directory.split()) 

library_dirs = []    

if args.opencv_libraries:
    library_dirs.extend(args.opencv_libraries.split()) 


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


#if "use_csiro_alignment" in unknown_args:
if args.use_csiro_alignment:
    #sys.argv.remove("use_csiro_alignment")
    os.environ["USE_CSIRO_ALIGNMENT"] = "1"
    extensions[2] = Extension("alignment", ["alignment.pyx", alignment_dir + "ForestBasedRegression.cc", alignment_dir + "Alignment.cc"],
                              language = "c++",
                              define_macros = [("USE_CSIRO_ALIGNMENT", 1)],
                              extra_compile_args = cxx_flags,
                              include_dirs = include_dirs,
                              libraries = opencv_libs + ["clmTracker"],
                              library_dirs = library_dirs)


#if "build_alignment_training" in unknown_args:
if args.build_alignment_training:
    #sys.argv.remove("build_alignment_training")
    extensions.append(Extension("tree_based_regression", ["tree_based_regression.pyx", alignment_dir + "ForestBasedRegression.cc"],
                                language = "c++",
                                extra_compile_args = cxx_flags + ["-fopenmp", "-ffast-math", "-O3", "-std=c++0x"],
                                extra_link_args = ["-fopenmp", "-ffast-math"],
                                include_dirs = include_dirs,
                                libraries = opencv_libs,
                                library_dirs = library_dirs)
                  )
    

setup(
    name = "descriptors",
    ext_modules = cythonize(extensions, gdb_debug=str(args.debug))
    )
