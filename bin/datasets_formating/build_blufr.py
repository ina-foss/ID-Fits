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
import argparse
import scipy.io
import numpy as np


execfile("fix_imports.py")
import config
from datasets import lfw



to_retrieve = {
    "labels": "labels",
    "devTrainIndex": "dev_train",
    "devTestIndex": "dev_test",
    "trainIndex": "train",
    "testIndex": "test",
    "galIndex": "gallery",
    "probIndex": "probe_images"
}



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Format BLUFR benchmark files")
    parser.add_argument("directory", help="BLUFR directory")
    args = parser.parse_args()

    mapping = lfw.loadMapping()
    data_file = os.path.join(args.directory, "config", "lfw", "blufr_lfw_config.mat")
    mat = scipy.io.loadmat(data_file)


    reindex = []
    for image_filename in mat["imageList"]:
        root = os.path.splitext(image_filename[0][0])[0]
        name = root[:-5]
        subindex = int(root[-4:])
        reindex.append(mapping[(name, subindex)])

    data = {"indexes": np.asarray(reindex)}
    for entry in mat:
        if entry in to_retrieve:
            data[to_retrieve[entry]] = mat[entry].ravel()
            for i, set_data in enumerate(data[to_retrieve[entry]]):
                data[to_retrieve[entry]][i] = set_data.ravel()

    np.savez(os.path.join(config.lfw_path, "blufr"), **data)
