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


import os


base_path = os.path.realpath(os.path.join(os.path.dirname(__file__), os.pardir))

bin_path = os.path.join(base_path, "bin")

models_path = os.path.join(base_path, "models")

data_path = os.path.join(base_path, "data")
lfw_path = os.path.join(data_path, "lfw")
bioid_path = os.path.join(data_path, "bioid")
_300w_path = os.path.join(data_path, "300-w")

benchmarks_path = os.path.join(base_path, "benchmarks")
lfw_benchmark_path = os.path.join(benchmarks_path, "lfw")
blufr_benchmark_path = os.path.join(benchmarks_path, "blufr")
_300w_benchmark_path = os.path.join(benchmarks_path, "300w")

descriptors_path = os.path.join(base_path, "descriptors")


