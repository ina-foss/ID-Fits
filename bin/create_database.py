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
import numpy as np

import fix_imports
import config
from datasets import lfw
from descriptors import *
from utils.file_manager import makedirsIfNeeded



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Builds database for IR.")
    parser.add_argument("descriptors", help="descriptor to use")
    args = parser.parse_args()

    filename = os.path.splitext(os.path.basename(args.descriptors))[0]

    mapping = lfw.loadMapping()
    labels = [""]*len(mapping)

    sets = lfw.loadTestSets()
    test_set = []
    for s in sets[:-1]:
        test_set += s
    
    descs = np.load(args.descriptors.strip())[test_set]

    for (name, _), index in mapping.iteritems():
        if index in test_set:
            labels[index] = name
    labels = np.asarray(labels)[test_set]

    output_file = os.path.join(config.databases_path, filename)
    makedirsIfNeeded(output_file)
    np.savez(output_file, descriptors=descs, labels=labels)
    print "Saved to %s.npz" % output_file

