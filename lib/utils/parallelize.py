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
import tempfile
import shutil
import numpy as np
from joblib import Parallel, delayed, load, dump


def parallelize(function, input_data, output_shape, output_dtype=None, args=None, output_file=None, n_jobs=2, cached=True, load_as_array=False):
    if not output_dtype:
        output_dtype = input_data.dtype

    if output_file and cached and os.path.exists(output_file):
        try:
            mmap = np.memmap(output_file, shape=output_shape, dtype=output_dtype, mode="c")
            if load_as_array:
                return np.array(mmap)
            else:
                return mmap
        except:
            print "Cannot open previously cached output file (maybe due to different arguments), recomputes from scratch"

    if output_file and not os.path.exists(os.path.split(output_file)[0]):
	os.makedirs(os.path.split(output_file)[0])

    temp_folder = tempfile.mkdtemp()
    temp_file = os.path.join(temp_folder, "input_data")
    if not output_file:
        output_file = os.path.join(temp_folder, "output_data")
	temp_output_file = True
    else:
	temp_output_file = False

    dump(input_data, temp_file)
    data = load(temp_file, mmap_mode='r')
    output_mmap = np.memmap(output_file, shape=output_shape, dtype=output_dtype, mode="w+")

    parallel = Parallel(n_jobs=n_jobs)
    if args:
        parallel(delayed(function)(data, output_mmap, i, *args) for i in range(len(data)))
    else:
        parallel(delayed(function)(data, output_mmap, i) for i in range(len(data)))

    output_mmap.flush()

    mmap = np.memmap(output_file, shape=output_shape, dtype=output_dtype, mode="c")
    if load_as_array or temp_output_file:
	mmap = np.array(mmap)

    shutil.rmtree(temp_folder)
    return mmap
